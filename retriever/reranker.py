class CompositionalReranker:
    """
    Reranks candidate fashion images by combining multi-modal similarity scores.

    Weights:
      - Global CLIP similarity: 0.30
      - Regional CLIP similarity: 0.40
      - Compositional color/garment metadata matching: 0.20
      - Scene classifier metadata matching: 0.10
    """

    def __init__(self, compositional_matcher, metadata=None):
        self.matcher = compositional_matcher
        self.metadata = metadata or {}

    def rerank(self, candidates, parsed_query, top_k=10):
        """
        Compute weighted final score and rank candidate images.

        Args:
            candidates: dict mapping image_id -> {"global_clip_score": float, "regional_clip_score": float}
            parsed_query: structured dict from query_parser
            top_k: number of final results to return

        Returns:
            list of dicts sorted descending by final score
        """
        results = []
        has_garments = bool(parsed_query.get("garments", []))

        for image_id, scores in candidates.items():
            global_s = scores.get("global_clip_score", 0.0)
            regional_s = scores.get("regional_clip_score", 0.0)
            comp_s = self.matcher.score_compositional(image_id, parsed_query)
            scene_s = self.matcher.score_scene(image_id, parsed_query)

            if has_garments:
                w_global = 0.30
                w_regional = 0.40
                w_comp = 0.20
                w_scene = 0.10
            else:
                # If query has no specific garment regional requirement, redistribute regional weight to global
                w_global = 0.60
                w_regional = 0.00
                w_comp = 0.25
                w_scene = 0.15

            final_score = (
                w_global * global_s +
                w_regional * regional_s +
                w_comp * comp_s +
                w_scene * scene_s
            )

            results.append({
                "path": image_id,
                "score": float(final_score),
                "global_clip_score": float(global_s),
                "regional_clip_score": float(regional_s),
                "compositional_score": float(comp_s),
                "scene_score": float(scene_s),
                "metadata": self.metadata.get(image_id, {}),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
