import numpy as np

from retriever.compositional_matcher import SCENE_ALIASES


class CandidateRetriever:
    """
    Retrieves and fuses candidate images from global and regional FAISS indices,
    with optional scene-based retrieval from metadata.
    """

    def __init__(self, vector_store, clip_encoder, metadata=None):
        self.store = vector_store
        self.clip = clip_encoder
        self.metadata = metadata or {}

    def search_global(self, query_text, k=50):
        """Search global FAISS index."""
        if "global" not in self.store.indices or self.store.count("global") == 0:
            return {}
        vec = self.clip.encode_text(query_text)
        scores, ids = self.store.search("global", vec, k=k)
        return {img_id: float(score) for img_id, score in zip(ids, scores)}

    def search_regional(self, garments, k=50):
        """Search regional garment FAISS index across all parsed garment queries.

        Returns:
            dict mapping image_id -> {
                "overall": float,           # max across all garment searches
                "per_garment": {label: float}  # best similarity per garment label
            }
        """
        if "regional" not in self.store.indices or self.store.count("regional") == 0 or not garments:
            return {}

        aggregated_scores = {}
        per_garment_best = {}

        for g in garments:
            desc = g.get("description") or g.get("label", "")
            label = g.get("label", "unknown")
            if not desc:
                continue
            vec = self.clip.encode_text(desc)
            scores, crop_ids = self.store.search("regional", vec, k=k)

            for crop_id, score in zip(crop_ids, scores):
                img_id = crop_id.split("#crop_")[0] if "#crop_" in crop_id else crop_id
                score = float(score)

                if img_id not in aggregated_scores or score > aggregated_scores[img_id]:
                    aggregated_scores[img_id] = score

                if img_id not in per_garment_best:
                    per_garment_best[img_id] = {}
                if label not in per_garment_best[img_id] or score > per_garment_best[img_id][label]:
                    per_garment_best[img_id][label] = score

        result = {}
        for img_id in aggregated_scores:
            result[img_id] = {
                "overall": aggregated_scores[img_id],
                "per_garment": per_garment_best.get(img_id, {}),
            }

        return result

    def search_by_scene(self, scene_label, k=50):
        """Retrieve candidates whose metadata scene_probs match the target scene.

        Unlike global/regional FAISS search, this queries the metadata directly
        to ensure scene-relevant images enter the candidate pool even when CLIP
        text search does not surface them.

        Returns:
            dict mapping image_id -> scene_score (sum of matching probabilities)
        """
        if not scene_label or not self.metadata:
            return {}

        aliases = SCENE_ALIASES.get(scene_label, set())
        if not aliases:
            return {}

        matches = []
        for image_id, meta in self.metadata.items():
            scene_probs = meta.get("scene_probs", {})
            score = 0.0
            for cat, prob in scene_probs.items():
                normalized = cat.lower().replace(" ", "_")
                base = normalized.split("/")[0]
                if base in aliases:
                    score += prob
            if score > 0:
                matches.append((image_id, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return {img_id: score for img_id, score in matches[:k]}

    def _backfill_global_scores(self, candidates, query_vec):
        """Compute global CLIP similarity for candidates missing a global score.

        Uses FAISS reconstruct to retrieve stored vectors and compute cosine similarity.
        """
        missing = [img_id for img_id, c in candidates.items() if c["global_clip_score"] == 0.0]
        if not missing or "global" not in self.store.indices:
            return

        id_map = self.store.id_maps.get("global", [])
        index = self.store.indices["global"]

        for img_id in missing:
            try:
                idx = id_map.index(img_id)
                stored_vec = index.reconstruct(idx)
                sim = float(np.dot(query_vec.flatten(), stored_vec.flatten()))
                candidates[img_id]["global_clip_score"] = sim
            except (ValueError, RuntimeError):
                continue

    def get_candidates(self, parsed_query, raw_query, k=50):
        """
        Perform candidate retrieval via global, regional, and scene search and fuse them.

        Returns:
            dict mapping image_id -> {
                "global_clip_score": float,
                "regional_clip_score": float,
                "regional_per_garment": {label: float},
            }
        """
        global_text = parsed_query.get("global_refined_query") or raw_query

        # Relationship preservation: if refined query dropped significant words,
        # fall back to the raw query to preserve spatial/action context
        if len(global_text.split()) < len(raw_query.split()) * 0.5:
            global_text = raw_query

        global_vec = self.clip.encode_text(global_text)
        global_results = self.search_global(global_text, k=k)
        regional_results = self.search_regional(parsed_query.get("garments", []), k=k)

        # Scene-based retrieval: inject candidates matching the target scene
        scene_label = parsed_query.get("scene", {}).get("label")
        scene_results = self.search_by_scene(scene_label, k=k) if scene_label else {}

        all_ids = set(global_results.keys()).union(regional_results.keys()).union(scene_results.keys())
        candidates = {}

        for img_id in all_ids:
            reg = regional_results.get(img_id, {})
            candidates[img_id] = {
                "global_clip_score": global_results.get(img_id, 0.0),
                "regional_clip_score": reg.get("overall", 0.0),
                "regional_per_garment": reg.get("per_garment", {}),
            }

        self._backfill_global_scores(candidates, global_vec)

        return candidates
