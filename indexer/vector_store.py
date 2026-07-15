import os
import json
import numpy as np
import faiss


class VectorStore:
    """
    Manages FAISS indices for global and regional embeddings.

    Each named index stores vectors of a fixed dimension with associated IDs
    (image paths or regional crop IDs like `path#crop_0`).
    Supports adding, searching, saving, and loading indices and ID mappings.
    """

    def __init__(self, save_dir=None):
        self.indices = {}
        self.id_maps = {}
        self.save_dir = save_dir

    def add_index(self, name, dim, metric="ip"):
        """
        Create and register a new FAISS index.

        Args:
            name: Index identifier (e.g. "global", "regional")
            dim: Vector dimensionality (e.g. 768 for ViT-L/14)
            metric: Distance metric ("ip" for inner product / cosine, "l2" for Euclidean)
        """
        if metric == "ip":
            index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        self.indices[name] = index
        self.id_maps[name] = []

    def add_vectors(self, name, vectors, ids):
        """
        Add vectors and their associated IDs to a named index.

        Args:
            name: Index name ("global" or "regional")
            vectors: np.ndarray shape (N, D) float32
            ids: list of str IDs
        """
        if name not in self.indices:
            raise ValueError(f"Index '{name}' not found. Call add_index first.")

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        self.indices[name].add(vectors)
        self.id_maps[name].extend(ids)

    def search(self, name, query, k=10):
        """
        Search a named index for the top-k nearest vectors.

        Args:
            name: Index name ("global" or "regional")
            query: np.ndarray shape (D,) or (1, D) float32
            k: Number of nearest neighbors to retrieve

        Returns:
            scores: np.ndarray shape (k,) similarity scores
            ids: list of k str IDs
        """
        if name not in self.indices or self.indices[name].ntotal == 0:
            return np.array([]), []

        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        faiss.normalize_L2(query)

        k_search = min(k, self.indices[name].ntotal)
        if k_search == 0:
            return np.array([]), []

        scores, indices = self.indices[name].search(query, k_search)
        scores = scores.flatten()
        ids = [self.id_maps[name][i] for i in indices.flatten()]
        return scores, ids

    def count(self, name):
        """Return the total number of vectors in the named index."""
        if name not in self.indices:
            return 0
        return self.indices[name].ntotal

    def save(self, directory=None):
        """Save all FAISS indices and ID maps to disk."""
        directory = directory or self.save_dir
        if directory is None:
            raise ValueError("No save directory specified.")
        os.makedirs(directory, exist_ok=True)

        for name, index in self.indices.items():
            faiss.write_index(index, os.path.join(directory, f"{name}.index"))
            with open(os.path.join(directory, f"{name}_ids.json"), "w", encoding="utf-8") as f:
                json.dump(self.id_maps[name], f, indent=2)

    def load(self, directory=None):
        """Load FAISS indices and ID maps from disk."""
        directory = directory or self.save_dir
        if directory is None or not os.path.exists(directory):
            raise ValueError(f"Save directory {directory} does not exist.")

        for fname in os.listdir(directory):
            if fname.endswith(".index"):
                name = fname.replace(".index", "")
                index = faiss.read_index(os.path.join(directory, fname))
                self.indices[name] = index
                ids_path = os.path.join(directory, f"{name}_ids.json")
                if os.path.exists(ids_path):
                    with open(ids_path, "r", encoding="utf-8") as f:
                        self.id_maps[name] = json.load(f)
                else:
                    self.id_maps[name] = []
