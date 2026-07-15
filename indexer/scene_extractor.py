import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as trn
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights", "places365")

# Fallback check
if not os.path.exists(WEIGHTS_DIR):
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "environment", "weights")


def _load_wideresnet():
    if WEIGHTS_DIR not in sys.path:
        sys.path.insert(0, WEIGHTS_DIR)
    import wideresnet
    return wideresnet


def load_scene_categories():
    path = os.path.join(WEIGHTS_DIR, "categories_places365.txt")
    classes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            classes.append(line.strip().split(" ")[0][3:])
    return tuple(classes)


def load_io_labels():
    path = os.path.join(WEIGHTS_DIR, "IO_places365.txt")
    with open(path, "r", encoding="utf-8") as f:
        labels = [int(line.strip().split()[-1]) - 1 for line in f]
    return np.array(labels)


def load_scene_attributes():
    path = os.path.join(WEIGHTS_DIR, "labels_sunattribute.txt")
    with open(path, "r", encoding="utf-8") as f:
        attrs = [line.strip() for line in f if line.strip()]
    return attrs


def load_attribute_weights():
    path = os.path.join(WEIGHTS_DIR, "W_sceneattribute_wideresnet18.npy")
    return np.load(path)


class SceneExtractor:
    """
    Extracts scene context from images using Places365 (WideResNet18).

    Designed to operate on cleaned environment background images where person and
    garment bounding boxes have been masked out.
    """

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        wideresnet = _load_wideresnet()
        model = wideresnet.resnet18(num_classes=365)

        weight_path = os.path.join(WEIGHTS_DIR, "wideresnet18_places365.pth.tar")
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.model = model

        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.categories = load_scene_categories()
        self.io_labels = load_io_labels()
        self.attributes = load_scene_attributes()
        self.attr_weights = load_attribute_weights()

    def _preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # If BGR, convert to RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1]).convert("RGB")
            else:
                image = Image.fromarray(image).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, image):
        """
        Extract scene features from an image or cleaned background array.

        Returns:
            dict with scene_embedding (512,), scene_probs, scene_category,
            indoor_outdoor, scene_attributes, attribute_features
        """
        input_tensor = self._preprocess(image)
        features_blob = []

        def hook(module, input, output):
            features_blob.append(output.data.cpu().numpy())

        avgpool = dict(self.model.named_modules())["avgpool"]
        handle = avgpool.register_forward_hook(hook)

        logit = self.model(input_tensor)
        handle.remove()

        probs = F.softmax(logit, dim=1).data.squeeze().cpu().numpy()
        sorted_indices = np.argsort(probs)[::-1]

        scene_embedding = features_blob[0].squeeze()
        scene_embedding = scene_embedding / (np.linalg.norm(scene_embedding) + 1e-8)

        top5 = {
            self.categories[sorted_indices[i]]: float(probs[sorted_indices[i]])
            for i in range(min(5, len(sorted_indices)))
        }

        io_pred = np.mean(self.io_labels[sorted_indices[:10]])
        indoor_outdoor = "indoor" if io_pred < 0.5 else "outdoor"

        attr_features = self.attr_weights.dot(scene_embedding)
        attr_indices = np.argsort(attr_features)[::-1]
        top_attrs = [self.attributes[attr_indices[i]] for i in range(min(10, len(attr_indices)))]

        raw_attr = attr_features / (np.linalg.norm(attr_features) + 1e-8)

        return {
            "scene_embedding": scene_embedding.astype(np.float32).tolist(),
            "scene_probs": top5,
            "scene_category": self.categories[sorted_indices[0]],
            "indoor_outdoor": indoor_outdoor,
            "scene_attributes": top_attrs,
            "attribute_features": raw_attr.astype(np.float32).tolist(),
        }
