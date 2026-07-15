import torch
import numpy as np
import clip
from PIL import Image


class CLIPEncoder:
    """
    Extracts global and regional image/text embeddings using OpenAI CLIP.

    Uses ViT-L/14 for high-quality 768-dimensional embeddings.
    Both images (full fashion image or cropped garment regions) and text queries
    are projected into a shared semantic vector space.
    """

    def __init__(self, model_name="ViT-L/14", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.embedding_dim = 768 if "L/14" in model_name else 512

    def _prepare_pil(self, image):
        """Convert path or numpy array to RGB PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # If BGR from OpenCV (3 channels), convert to RGB
            if image.ndim == 3 and image.shape[2] == 3:
                return Image.fromarray(image[:, :, ::-1]).convert("RGB")
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def encode_image(self, image):
        """
        Extract a normalized embedding from a full image or regional crop.

        Args:
            image: file path (str), PIL Image, or numpy BGR/RGB array

        Returns:
            np.ndarray shape (768,) float32 L2-normalized vector
        """
        pil_img = self._prepare_pil(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text):
        """
        Extract normalized embeddings from text query strings.

        Args:
            text: string or list of strings

        Returns:
            np.ndarray shape (N, 768) float32 L2-normalized vectors
        """
        if isinstance(text, str):
            text = [text]
        tokens = clip.tokenize(text, truncate=True).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        """
        Batch encode multiple images or garment crops.

        Args:
            images: list of paths, PIL Images, or numpy arrays
            batch_size: batch size for forward passes

        Returns:
            np.ndarray shape (N, 768) float32 L2-normalized vectors
        """
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = [self.preprocess(self._prepare_pil(img)) for img in images[i : i + batch_size]]
            batch_tensor = torch.stack(batch).to(self.device)
            features = self.model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        if not all_features:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(all_features).astype(np.float32)

    def similarity(self, image_features, text_features):
        """
        Compute cosine similarity matrix between image and text features.
        Assumes inputs are already L2-normalized.
        """
        if image_features.ndim == 1:
            image_features = image_features.reshape(1, -1)
        if text_features.ndim == 1:
            text_features = text_features.reshape(1, -1)
        return image_features @ text_features.T
