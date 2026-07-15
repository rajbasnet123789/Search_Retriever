class CompositionalMatcher:
    """
    Computes compositional matching scores between parsed query attributes
    and indexed image metadata (garment labels, colors, and scene attributes).
    """

    def __init__(self, metadata=None):
        self.metadata = metadata or {}

    def score_compositional(self, image_id, parsed_query):
        """
        Evaluate how well the image matches required garment labels and colors.

        Returns:
            float between 0.0 and 1.0
        """
        target_garments = parsed_query.get("garments", [])
        if not target_garments:
            return 0.5

        meta = self.metadata.get(image_id, {})
        detected_garments = meta.get("garments", [])
        full_primary = meta.get("primary_color", "")
        full_dominants = meta.get("dominant_colors", [])

        if not detected_garments:
            # Fallback to overall image color matching if no crops detected
            matched_colors = 0
            for g in target_garments:
                req_color = g.get("color", "").lower()
                if req_color and (req_color == full_primary or req_color in full_dominants):
                    matched_colors += 1
            return min(1.0, matched_colors / len(target_garments)) if target_garments else 0.5

        score_sum = 0.0
        for target in target_garments:
            req_label = target.get("label", "").lower()
            req_color = target.get("color", "").lower()

            best_match_score = 0.0
            for det in detected_garments:
                det_label = det.get("label", "").lower()
                det_primary = det.get("primary_color", "").lower()
                det_dominants = [c.lower() for c in det.get("dominant_colors", [])]

                label_match = (
                    req_label in det_label or det_label in req_label or
                    any(word in det_label for word in req_label.split(",")) or
                    any(word in req_label for word in det_label.split(","))
                )

                color_match = (
                    not req_color or req_color == "unknown" or req_color == "neutral" or
                    req_color == det_primary or req_color in det_dominants
                )

                if label_match and color_match:
                    best_match_score = max(best_match_score, 1.0)
                elif label_match:
                    best_match_score = max(best_match_score, 0.6)
                elif color_match:
                    best_match_score = max(best_match_score, 0.3)

            score_sum += best_match_score

        return min(1.0, score_sum / len(target_garments))

    def score_scene(self, image_id, parsed_query):
        """
        Evaluate how well the image matches required scene attributes and formality.

        Returns:
            float between 0.0 and 1.0
        """
        scene_info = parsed_query.get("scene", {})
        target_label = scene_info.get("label", "").lower()
        formality = scene_info.get("formality")

        if not target_label and not formality:
            return 0.5

        meta = self.metadata.get(image_id, {})
        cat = meta.get("scene_category", "").lower()
        attrs = [a.lower() for a in meta.get("scene_attributes", [])]
        io = meta.get("indoor_outdoor", "").lower()

        score = 0.0
        if target_label and target_label != "general":
            if target_label in cat or cat in target_label:
                score += 0.6
            for attr in attrs:
                if target_label in attr:
                    score += 0.3
                    break

        if formality == "formal" and io == "indoor":
            score += 0.1
        elif formality == "casual" and io == "outdoor":
            score += 0.1
        elif formality == "sporty" and ("gym" in cat or "athletic" in cat or any("sport" in a for a in attrs)):
            score += 0.3

        return min(1.0, score if score > 0 else 0.4)
