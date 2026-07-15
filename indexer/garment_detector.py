import os
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FASHION_WEIGHTS = os.path.join(ROOT_DIR, "weights", "yolo", "best (1).pt")
DEFAULT_PERSON_WEIGHTS = os.path.join(ROOT_DIR, "weights", "yolo", "yolov8n.pt")


class GarmentDetector:
    """
    YOLO-based garment regional detector and environment background extractor.

    1. Uses Fashionpedia YOLO (best (1).pt) to detect and crop garment regions.
    2. Uses COCO YOLO (yolov8n.pt) for 'person' class detection.
    3. Masks out both 'person' and fashion detections so the remaining image
       represents the clean background/environment for Places365 scene extraction.
    """

    def __init__(self, fashion_model_path=None, person_model_path=None):
        fashion_path = fashion_model_path or DEFAULT_FASHION_WEIGHTS
        person_path = person_model_path or DEFAULT_PERSON_WEIGHTS

        if not os.path.exists(fashion_path):
            # Fallback check
            fallback = os.path.join(ROOT_DIR, "IMAGE_DETECTOR", "runs", "COLAB_WEIGHTS", "best (1).pt")
            if os.path.exists(fallback):
                fashion_path = fallback
            else:
                raise FileNotFoundError(f"Fashion YOLO weights not found at {fashion_path}")

        if not os.path.exists(person_path):
            fallback = os.path.join(ROOT_DIR, "IMAGE_DETECTOR", "yolov8n.pt")
            if os.path.exists(fallback):
                person_path = fallback

        self.fashion_model = YOLO(fashion_path)
        self.person_model = YOLO(person_path) if os.path.exists(person_path) else None

    def _load_img(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image from path: {image}")
            return img
        elif isinstance(image, np.ndarray):
            return image
        raise ValueError(f"Unsupported image format: {type(image)}")

    def detect_garments(self, image, conf=0.25):
        """
        Detect garment regions in an image and crop them.

        Args:
            image: file path (str) or numpy BGR array
            conf: confidence threshold

        Returns:
            list of dicts, each containing:
                - box_idx: integer index
                - box: [x1, y1, x2, y2] int bounding box coordinates
                - class_id: int class ID
                - label: string class label (e.g., 'shirt', 'pants', 'dress', 'shoe')
                - confidence: float detection confidence
                - crop: numpy BGR array of the cropped box
        """
        img = self._load_img(image)
        results = self.fashion_model(img, conf=conf, verbose=False)
        detections = []

        if not results or not results[0].boxes:
            return detections

        res = results[0]
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            label = res.names[cls_id]

            crop = img[y1:y2, x1:x2].copy()
            detections.append({
                "box_idx": i,
                "box": [x1, y1, x2, y2],
                "class_id": cls_id,
                "label": label,
                "confidence": conf_val,
                "crop": crop,
            })

        return detections

    def extract_environment_background(self, image, conf=0.25, fill_color=(255, 255, 255)):
        """
        Mask out 'person' class and all fashion boxes to extract clean environment/background.

        Args:
            image: file path (str) or numpy BGR array
            conf: confidence threshold
            fill_color: RGB/BGR tuple to fill masked areas (default white [255, 255, 255])

        Returns:
            numpy BGR array of the cleaned environment background
        """
        img = self._load_img(image)
        bg = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 1. Mask out person class
        if self.person_model:
            person_res = self.person_model(img, conf=conf, verbose=False)
            if person_res and person_res[0].boxes:
                for box in person_res[0].boxes:
                    cls_id = int(box.cls[0])
                    label = person_res[0].names[cls_id]
                    if label.lower() == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                        mask[y1:y2, x1:x2] = 255

        # 2. Mask out fashion dataset classes
        fashion_res = self.fashion_model(img, conf=conf, verbose=False)
        if fashion_res and fashion_res[0].boxes:
            for box in fashion_res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                mask[y1:y2, x1:x2] = 255

        # Replace masked regions so only background/environment remains
        bg[mask == 255] = fill_color
        return bg

    def process_image(self, image, conf=0.25):
        """
        Run both garment detection/cropping and environment background extraction in one pass.

        Returns:
            tuple: (garment_detections_list, background_bgr_array)
        """
        img = self._load_img(image)
        garment_crops = self.detect_garments(img, conf=conf)
        background_img = self.extract_environment_background(img, conf=conf)
        return garment_crops, background_img
