import cv2
import numpy as np
from sklearn.cluster import KMeans

COLOR_NAMES = {
    (0, 10): "red",
    (10, 25): "orange",
    (25, 40): "yellow",
    (40, 80): "green",
    (80, 130): "blue",
    (130, 170): "purple",
    (170, 180): "red",
}

NEUTRAL_THRESHOLDS = {
    "black": (0, 0, 0, 360, 0, 50),
    "white": (0, 0, 200, 360, 30, 255),
    "gray": (0, 0, 50, 360, 50, 200),
    "beige": (15, 40, 140, 30, 100, 240),
    "brown": (8, 25, 50, 150, 50, 150),
    "pink": (160, 175, 40, 150, 150, 255),
}


def hsv_to_color_name(h, s, v):
    """Map a single HSV value (OpenCV range H:0-180, S:0-255, V:0-255) to a color name."""
    if s < 30:
        if v < 50:
            return "black"
        if v > 200:
            return "white"
        return "gray"
    if v < 40:
        return "black"
    for name, (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) in NEUTRAL_THRESHOLDS.items():
        if h_lo <= h <= h_hi and s_lo <= s <= s_hi and v_lo <= v <= v_hi:
            return name
    h_mod = h % 180
    for (lo, hi), name in COLOR_NAMES.items():
        if lo <= h_mod < hi:
            return name
    return "unknown"


class ColorExtractor:
    """
    Extracts dominant HSV colors and color histograms from images or garment crops.
    """

    def __init__(self, n_dominant=5, hist_bins=(36, 10, 10)):
        self.n_dominant = n_dominant
        self.h_bins, self.s_bins, self.v_bins = hist_bins

    def _load_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        elif isinstance(image, np.ndarray):
            # If RGB, convert to BGR for cv2.COLOR_BGR2HSV
            return image
        raise ValueError(f"Unsupported image type: {type(image)}")

    def _get_dominant_colors(self, hsv_pixels, weights=None):
        if len(hsv_pixels) == 0:
            return np.zeros((self.n_dominant, 3)), np.zeros(self.n_dominant)

        n_clusters = min(self.n_dominant, len(hsv_pixels))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(hsv_pixels, sample_weight=weights)
        centers = kmeans.cluster_centers_

        counts = np.bincount(labels, minlength=n_clusters)
        proportions = counts / counts.sum()

        order = np.argsort(proportions)[::-1]
        centers = centers[order]
        proportions = proportions[order]

        if len(centers) < self.n_dominant:
            pad = np.zeros((self.n_dominant - len(centers), 3))
            centers = np.vstack([centers, pad])
            proportions = np.concatenate([proportions, np.zeros(self.n_dominant - len(proportions))])

        return centers, proportions

    def _get_histograms(self, hsv_image):
        h = hsv_image[:, :, 0].flatten()
        s = hsv_image[:, :, 1].flatten()
        v = hsv_image[:, :, 2].flatten()

        h_hist, _ = np.histogram(h, bins=self.h_bins, range=(0, 180))
        s_hist, _ = np.histogram(s, bins=self.s_bins, range=(0, 256))
        v_hist, _ = np.histogram(v, bins=self.v_bins, range=(0, 256))

        h_hist = h_hist.astype(np.float32) / (h_hist.sum() + 1e-8)
        s_hist = s_hist.astype(np.float32) / (s_hist.sum() + 1e-8)
        v_hist = v_hist.astype(np.float32) / (v_hist.sum() + 1e-8)

        return np.concatenate([h_hist, s_hist, v_hist])

    def extract(self, image):
        """
        Extract color features from a garment crop or full image.

        Returns:
            dict with dominant_colors, dominant_proportions, dominant_names,
            histogram, feature_vector, primary_color
        """
        img = self._load_image(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape(-1, 3).astype(np.float32)
        mask = (hsv[:, :, 1] > 10).flatten()
        valid_pixels = pixels[mask]
        if len(valid_pixels) == 0:
            valid_pixels = pixels

        centers, proportions = self._get_dominant_colors(valid_pixels)
        names = [hsv_to_color_name(c[0], c[1], c[2]) for c in centers]
        hist = self._get_histograms(hsv)

        feature = np.concatenate([
            centers.flatten() / np.tile([180.0, 255.0, 255.0], self.n_dominant),
            proportions,
            hist,
        ]).astype(np.float32)

        primary = names[0] if names else "unknown"

        return {
            "dominant_colors": centers.tolist(),
            "dominant_proportions": proportions.tolist(),
            "dominant_names": names,
            "histogram": hist.tolist(),
            "feature_vector": feature.tolist(),
            "primary_color": primary,
        }
