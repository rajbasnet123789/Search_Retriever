def _classify_query(parsed_query):
    """Classify the query type for weight selection.

    Returns one of: "compositional", "garment_scene", "garment", "scene", "style".
    """
    garments = parsed_query.get("garments", [])
    scene = parsed_query.get("scene", {})
    formality = scene.get("formality")
    scene_label = scene.get("label")

    n_garments = len(garments)
    has_scene = bool(scene_label)

    if n_garments >= 2:
        return "compositional"
    if n_garments == 1 and has_scene:
        return "garment_scene"
    if n_garments >= 1:
        return "garment"
    if formality and has_scene:
        return "style"
    if has_scene:
        return "scene"
    return "style"


_WEIGHT_PROFILES = {
    "compositional": {"global": 0.15, "regional": 0.35, "composition": 0.45, "scene": 0.05},
    "garment_scene": {"global": 0.20, "regional": 0.25, "composition": 0.30, "scene": 0.25},
    "garment":       {"global": 0.20, "regional": 0.40, "composition": 0.35, "scene": 0.05},
    "scene":         {"global": 0.40, "regional": 0.10, "composition": 0.15, "scene": 0.35},
    "style":         {"global": 0.55, "regional": 0.10, "composition": 0.10, "scene": 0.25},
}


def _tier_score(parsed_query, image_id, matcher):
    """Compute tier bonus based on explicit garment coverage.

    Tier 1 (all explicit garments present): 1.0
    Tier 2 (some present): 0.7
    Tier 3 (none present): 0.4
    """
    explicit_labels = [
        g["label"] for g in parsed_query.get("garments", [])
        if g.get("explicit") and g.get("label")
    ]
    if not explicit_labels:
        return 1.0

    present = sum(1 for label in explicit_labels if matcher.has_garment(image_id, label))
    ratio = present / len(explicit_labels)

    if ratio >= 1.0:
        return 1.0
    if ratio > 0.0:
        return 0.7
    return 0.4


class CompositionalReranker:
    """
    Reranks candidate fashion images by combining multi-modal similarity scores
    with query-dependent weighting and progressive tier fallback.
    """

    def __init__(self, compositional_matcher, metadata=None):
        self.matcher = compositional_matcher
        self.metadata = metadata or {}

    def rerank(self, candidates, parsed_query, top_k=10):
        """
        Compute weighted final score and rank candidate images.

        Uses tier-based progressive fallback: candidates matching all explicit
        garments rank above partial matches, which rank above global-only matches.

        Args:
            candidates: dict mapping image_id -> {global_clip_score, regional_clip_score, regional_per_garment}
            parsed_query: structured dict from query_parser
            top_k: number of final results to return

        Returns:
            list of dicts sorted descending by final score
        """
        query_type = _classify_query(parsed_query)
        weights = _WEIGHT_PROFILES[query_type]

        results = []

        for image_id, scores in candidates.items():
            global_s = scores.get("global_clip_score", 0.0)
            comp_s, any_explicit_missing = self.matcher.score_compositional(image_id, parsed_query)
            scene_s = self.matcher.score_scene(image_id, parsed_query)
            regional_s = self._compute_regional_score(scores, parsed_query)

            base_score = (
                weights["global"] * global_s +
                weights["regional"] * regional_s +
                weights["composition"] * comp_s +
                weights["scene"] * scene_s
            )

            # Scene penalty: if query requires an explicit scene and candidate
            # has no scene match, penalize to push it below scene-matching candidates
            explicit_scene = parsed_query.get("scene", {}).get("label")
            if explicit_scene and scene_s == 0.0:
                base_score *= 0.55

            tier = _tier_score(parsed_query, image_id, self.matcher)
            final_score = base_score * tier

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

    def _compute_regional_score(self, scores, parsed_query):
        """Compute completeness-weighted regional score from per-garment similarities."""
        garments = parsed_query.get("garments", [])
        if not garments:
            return scores.get("regional_clip_score", 0.0)

        regional_per_garment = scores.get("regional_per_garment", {})
        if not regional_per_garment:
            return scores.get("regional_clip_score", 0.0)

        per_garment_scores = []
        for g in garments:
            label = g.get("label", "")
            best = regional_per_garment.get(label, 0.0)
            per_garment_scores.append(best)

        if not per_garment_scores:
            return 0.0

        matched = sum(s > 0 for s in per_garment_scores)
        completeness = matched / len(per_garment_scores)
        avg = sum(per_garment_scores) / len(per_garment_scores)

        return avg * completeness
