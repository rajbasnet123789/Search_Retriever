import os
import json
import numpy as np

from indexer.clip_encoder import CLIPEncoder
from indexer.garment_detector import GarmentDetector
from indexer.color_extractor import ColorExtractor
from indexer.scene_extractor import SceneExtractor
from indexer.vector_store import VectorStore


class FashionIndexer:
    """
    Orchestrates the hybrid global-regional indexing pipeline (Part A).

    Steps:
      1. Global CLIP Image Encoder -> Global Vector -> Global FAISS Index.
      2. Garment Detector (YOLO best (1).pt + yolov8n.pt) -> Garment Regions & Clean Background.
      3. Regional CLIP Embeddings + HSV Color Extraction from Garment Crops -> Regional FAISS Index & Metadata.
      4. Places365 Scene Classifier run on clean background -> Scene Category/Attributes in Metadata.
    """

    def __init__(self, index_dir=None, device=None, conf=0.25):
        self.device = device
        self.conf = conf
        self.clip = CLIPEncoder(device=device)
        self.detector = GarmentDetector()
        self.color_ext = ColorExtractor()
        self.scene_ext = SceneExtractor(device=device)
        self.store = VectorStore(save_dir=index_dir)
        self.metadata = {}

        if index_dir and os.path.exists(index_dir):
            try:
                self.load(index_dir)
            except Exception as e:
                print(f"Warning: could not load existing index from {index_dir}: {e}")

    def index_image(self, image_path):
        """
        Extract and index global, regional, color, and scene features for a single image.

        Returns:
            dict containing full structured metadata of the image.
        """
        # 1. Global CLIP vector
        clip_global = self.clip.encode_image(image_path)
        if "global" not in self.store.indices:
            self.store.add_index("global", dim=self.clip.embedding_dim, metric="ip")
        self.store.add_vectors("global", clip_global.reshape(1, -1), [image_path])

        # 2. Garment detection and clean environment background extraction
        garment_crops, clean_bg = self.detector.process_image(image_path, conf=self.conf)

        # 3. Scene extraction on clean environment (person + garments masked out)
        scene_res = self.scene_ext.extract(clean_bg)

        # 4. Global color extraction
        full_color = self.color_ext.extract(image_path)

        # 5. Regional garment processing
        garments_metadata = []
        if "regional" not in self.store.indices:
            self.store.add_index("regional", dim=self.clip.embedding_dim, metric="ip")

        for crop_info in garment_crops:
            crop_img = crop_info["crop"]
            crop_id = f"{image_path}#crop_{crop_info['box_idx']}"

            # Regional CLIP embedding
            clip_regional = self.clip.encode_image(crop_img)
            self.store.add_vectors("regional", clip_regional.reshape(1, -1), [crop_id])

            # Regional color extraction
            crop_color = self.color_ext.extract(crop_img)

            garments_metadata.append({
                "box_idx": crop_info["box_idx"],
                "crop_id": crop_id,
                "box": crop_info["box"],
                "label": crop_info["label"],
                "confidence": float(crop_info["confidence"]),
                "primary_color": crop_color["primary_color"],
                "dominant_colors": crop_color["dominant_names"],
                "dominant_proportions": crop_color["dominant_proportions"],
            })

        # Save comprehensive metadata entry
        meta = {
            "path": image_path,
            "global_clip_id": image_path,
            "scene_category": scene_res["scene_category"],
            "scene_probs": scene_res["scene_probs"],
            "indoor_outdoor": scene_res["indoor_outdoor"],
            "scene_attributes": scene_res["scene_attributes"],
            "primary_color": full_color["primary_color"],
            "dominant_colors": full_color["dominant_names"],
            "dominant_proportions": full_color["dominant_proportions"],
            "garments": garments_metadata,
        }
        self.metadata[image_path] = meta
        return meta

    def index_batch(self, image_paths, callback=None):
        """Index a list of image paths."""
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            try:
                self.index_image(path)
            except Exception as e:
                print(f"  Skip {os.path.basename(path)}: {e}")
            if callback:
                callback(i + 1, total)

    def save(self, directory=None):
        """Save FAISS indices and JSON metadata."""
        directory = directory or self.store.save_dir
        if directory is None:
            raise ValueError("No save directory specified.")
        os.makedirs(directory, exist_ok=True)
        self.store.save(directory)
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, directory=None):
        """Load FAISS indices and JSON metadata."""
        directory = directory or self.store.save_dir
        if directory is None or not os.path.exists(directory):
            raise ValueError(f"Save directory {directory} does not exist.")
        self.store.load(directory)
        meta_path = os.path.join(directory, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
