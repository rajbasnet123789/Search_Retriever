import numpy as np


class CandidateRetriever:
    """
    Retrieves and fuses candidate images from global and regional FAISS indices.
    """

    def __init__(self, vector_store, clip_encoder):
        self.store = vector_store
        self.clip = clip_encoder

    def search_global(self, query_text, k=50):
        """Search global FAISS index."""
        if "global" not in self.store.indices or self.store.count("global") == 0:
            return {}
        vec = self.clip.encode_text(query_text)
        scores, ids = self.store.search("global", vec, k=k)
        return {img_id: float(score) for img_id, score in zip(ids, scores)}

    def search_regional(self, garments, k=50):
        """Search regional garment FAISS index across all parsed garment queries."""
        if "regional" not in self.store.indices or self.store.count("regional") == 0 or not garments:
            return {}

        aggregated_scores = {}
        for g in garments:
            desc = g.get("description") or g.get("label", "")
            if not desc:
                continue
            vec = self.clip.encode_text(desc)
            scores, crop_ids = self.store.search("regional", vec, k=k)

            for crop_id, score in zip(crop_ids, scores):
                # Extract original parent image ID from crop ID (e.g. `path#crop_0` -> `path`)
                img_id = crop_id.split("#crop_")[0] if "#crop_" in crop_id else crop_id
                # Keep max similarity if multiple crops match
                if img_id not in aggregated_scores or score > aggregated_scores[img_id]:
                    aggregated_scores[img_id] = float(score)

        return aggregated_scores

    def get_candidates(self, parsed_query, raw_query, k=50):
        """
        Perform candidate retrieval via global and regional search and fuse them.

        Returns:
            dict mapping `image_id` to `{"global_clip_score": float, "regional_clip_score": float}`
        """
        global_text = parsed_query.get("global_refined_query") or raw_query
        global_results = self.search_global(global_text, k=k * 3)
        regional_results = self.search_regional(parsed_query.get("garments", []), k=k * 3)

        all_ids = set(global_results.keys()).union(regional_results.keys())
        candidates = {}

        for img_id in all_ids:
            candidates[img_id] = {
                "global_clip_score": global_results.get(img_id, 0.0),
                "regional_clip_score": regional_results.get(img_id, 0.0),
            }

        return candidates
