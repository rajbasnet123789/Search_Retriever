# ---------------------------------------------------------------------------
# Color matching – strict: primary colour first, then dominant list
# ---------------------------------------------------------------------------
def _color_match_score(requested, detected_primary, detected_dominants):
    """Grade the colour match between request and detection.

    Returns 0.0 – 1.0.
    Strict: only full credit for primary colour match.
    """
    if not requested or requested in ("unknown", "neutral"):
        return 1.0

    req = requested.lower()
    primary = detected_primary.lower()
    dominants = [d.lower() for d in detected_dominants]

    if req == primary:
        return 1.0
    if req in dominants:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Garment matching
# ---------------------------------------------------------------------------
def _label_match(requested_label, detected_label):
    """Check if a detected garment label satisfies the requested label."""
    req = requested_label.lower()
    det = detected_label.lower()
    if req == det:
        return True
    if req in det or det in req:
        return True
    req_words = {w.strip() for w in req.split(",")}
    det_words = {w.strip() for w in det.split(",")}
    if req_words & det_words:
        return True
    return False


def _garment_match_score(label_match, color_score):
    """Per-garment match score.

    Returns 0.0 – 1.0.
    """
    if label_match and color_score >= 0.8:
        return 1.0
    if label_match:
        return 0.4 + 0.4 * color_score
    return 0.0


# ---------------------------------------------------------------------------
# Scene matching
# ---------------------------------------------------------------------------
SCENE_ALIASES = {
    "park": {
        "park", "public_garden", "botanical_garden", "picnic_area",
        "playground", "garden", "flower_path",
    },
    "city": {
        "street", "downtown", "crosswalk", "promenade", "plaza",
        "shopping_street", "urban_area", "alley", "courtyard",
    },
    "office": {
        "office", "office_cubicles", "home_office", "conference_room",
    },
    "beach": {
        "beach", "sandbar", "lagoon", "ocean", "coast",
    },
    "home": {
        "home", "living_room", "bedroom", "kitchen", "bathroom",
        "home_theater", "den", "playroom",
    },
    "restaurant": {
        "restaurant", "cafeteria", "dining_room", "bar", "pub",
    },
    "cafe": {
        "cafe", "coffee_shop", "tea_room",
    },
    "gym": {
        "gym", "weight_room", "athletic_field", "recreation_room",
    },
    "store": {
        "store", "shop", "mall", "supermarket", "boutique",
    },
    "street": {
        "street", "crosswalk", "alley", "parking_lot",
    },
}


def _scene_match_score(target_label, scene_probs, indoor_outdoor):
    """Score scene compatibility using top-5 probabilities and alias sets.

    Returns 0.0 – 1.0.
    """
    if not target_label or target_label == "general":
        return 0.0

    aliases = SCENE_ALIASES.get(target_label, set())
    if not aliases:
        return 0.0

    score = 0.0
    for cat, prob in scene_probs.items():
        normalized = cat.lower().replace(" ", "_")
        base = normalized.split("/")[0]
        if base in aliases:
            score += prob

    return min(1.0, score)


def _formality_score(formality, indoor_outdoor, scene_probs):
    """Bonus for formality + scene alignment."""
    if not formality:
        return 0.0

    cats = {c.lower().replace(" ", "_") for c in scene_probs}

    if formality == "formal" and indoor_outdoor == "indoor":
        return 0.1
    if formality == "casual" and indoor_outdoor == "outdoor":
        return 0.1
    if formality == "sporty":
        sporty_cats = {"gym", "weight_room", "athletic_field", "recreation_room"}
        if cats & sporty_cats:
            return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# Formality garment alignment
# ---------------------------------------------------------------------------
FORMAL_GARMENTS = {
    "shirt, blouse", "jacket", "pants", "tie", "dress", "shoe",
    "cardigan", "vest", "skirt",
}
CASUAL_GARMENTS = {
    "top, t-shirt, sweatshirt", "shorts", "hood",
}


def _formality_garment_score(detected_garments, required_formality):
    """Score garment-based formality alignment.

    Returns a bonus (0.0-0.2) or penalty (-0.1) based on whether detected
    garments match the requested formality.
    """
    if not required_formality:
        return 0.0

    labels = {g.get("label", "").lower() for g in detected_garments}

    if required_formality == "formal":
        formal_count = len(labels & FORMAL_GARMENTS)
        casual_count = len(labels & CASUAL_GARMENTS)
        if formal_count > 0 and casual_count == 0:
            return 0.2
        if formal_count > 0:
            return 0.1
        return -0.1

    if required_formality == "casual":
        casual_count = len(labels & CASUAL_GARMENTS)
        if casual_count > 0:
            return 0.1
        return 0.0

    return 0.0


# ---------------------------------------------------------------------------
# Main compositional matcher
# ---------------------------------------------------------------------------
class CompositionalMatcher:
    """
    Computes compositional matching scores between parsed query attributes
    and indexed image metadata (garment labels, colors, and scene attributes).
    """

    def __init__(self, metadata=None):
        self.metadata = metadata or {}

    def score_compositional(self, image_id, parsed_query):
        """Evaluate how well the image matches required garment labels and colors.

        Uses one-to-one greedy assignment between requested garments and
        detected crops, with completeness penalty and confidence weighting.

        Returns:
            tuple(float, bool) -- (composition_score, any_explicit_missing)
            any_explicit_missing is True when an explicit garment had zero matches.
        """
        target_garments = parsed_query.get("garments", [])
        if not target_garments:
            return 0.0, False

        meta = self.metadata.get(image_id, {})
        detected_garments = meta.get("garments", [])

        if not detected_garments:
            return 0.0, True

        sorted_targets = sorted(
            target_garments,
            key=lambda t: (0 if t.get("color") and t["color"] not in ("unknown", "neutral") else 1),
        )

        used_crop_ids = set()
        per_garment_scores = []
        any_explicit_missing = False

        for target in sorted_targets:
            req_label = target.get("label", "")
            req_color = target.get("color", "")
            is_explicit = target.get("explicit", False)

            best_score = 0.0
            best_crop_id = None

            for det in detected_garments:
                crop_id = det.get("crop_id")
                if crop_id in used_crop_ids:
                    continue

                lbl_match = _label_match(req_label, det.get("label", ""))
                c_score = _color_match_score(
                    req_color,
                    det.get("primary_color", ""),
                    det.get("dominant_colors", []),
                )
                score = _garment_match_score(lbl_match, c_score)

                confidence = det.get("confidence", 1.0)
                score *= max(confidence, 0.35)

                if score > best_score:
                    best_score = score
                    best_crop_id = crop_id

            if best_crop_id:
                used_crop_ids.add(best_crop_id)

            if is_explicit and best_score == 0.0:
                any_explicit_missing = True

            per_garment_scores.append(best_score)

        if not per_garment_scores:
            return 0.0, True

        matched = sum(s > 0 for s in per_garment_scores)
        completeness = matched / len(per_garment_scores)
        avg_score = sum(per_garment_scores) / len(per_garment_scores)

        comp_score = avg_score * completeness

        if any_explicit_missing:
            comp_score *= 0.4

        return comp_score, any_explicit_missing

    def score_scene(self, image_id, parsed_query):
        """Evaluate how well the image matches required scene attributes.

        Returns:
            float between 0.0 and 1.0
        """
        scene_info = parsed_query.get("scene", {})
        target_label = scene_info.get("label", "")
        formality = scene_info.get("formality")

        if not target_label and not formality:
            return 0.0

        meta = self.metadata.get(image_id, {})
        scene_probs = meta.get("scene_probs", {})
        indoor_outdoor = meta.get("indoor_outdoor", "")
        detected_garments = meta.get("garments", [])

        scene_s = _scene_match_score(target_label, scene_probs, indoor_outdoor)
        form_s = _formality_score(formality, indoor_outdoor, scene_probs)

        # Only boost by garment formality when the scene actually matches
        formality_garment_s = 0.0
        if scene_s > 0:
            formality_garment_s = _formality_garment_score(detected_garments, formality)

        return min(1.0, max(0.0, scene_s + form_s + formality_garment_s))

    def has_garment(self, image_id, garment_label):
        """Check if an image has any crop matching the given garment label."""
        meta = self.metadata.get(image_id, {})
        for det in meta.get("garments", []):
            if _label_match(garment_label, det.get("label", "")):
                return True
        return False

    def required_garments(self, parsed_query):
        """Return set of garment labels required by the query."""
        return {g.get("label", "") for g in parsed_query.get("garments", []) if g.get("label")}
