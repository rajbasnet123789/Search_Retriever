import os
import re
import json
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


YOLO_FASHION_CLASSES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
    "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf", "umbrella", "hood", "collar"
]

COMMON_GARMENT_ALIASES = {
    "shirt": "shirt, blouse",
    "blouse": "shirt, blouse",
    "t-shirt": "top, t-shirt, sweatshirt",
    "tshirt": "top, t-shirt, sweatshirt",
    "tee": "top, t-shirt, sweatshirt",
    "top": "top, t-shirt, sweatshirt",
    "sweatshirt": "top, t-shirt, sweatshirt",
    "hoodie": "top, t-shirt, sweatshirt",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "blazer": "jacket",
    "raincoat": "coat",
    "coat": "coat",
    "parka": "coat",
    "pants": "pants",
    "trousers": "pants",
    "jeans": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "dress": "dress",
    "shoe": "shoe",
    "sneaker": "shoe",
    "boot": "shoe",
    "sandals": "shoe",
    "bag": "bag, wallet",
    "purse": "bag, wallet",
    "handbag": "bag, wallet",
    "backpack": "bag, wallet",
    "hat": "hat",
    "cap": "hat",
    "tie": "tie",
    "belt": "belt",
    "scarf": "scarf",
    "glasses": "glasses",
    "sunglasses": "glasses"
}

COLOR_KEYWORDS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "gray", "grey", "brown", "beige", "tan", "navy", "crimson", "olive"
]

SCENE_KEYWORDS = [
    "office", "street", "park", "home", "restaurant", "cafe", "gym", "beach", "store", "city", "urban"
]

SYSTEM_PROMPT = """You are an information extraction system for fashion image retrieval.

Extract garments, colours, scenes and attributes EXPLICITLY mentioned in the user's query.

CRITICAL RULES:
- Only extract garments that are directly named in the query.
- Never invent, infer or suggest garments, colours or accessories not explicitly stated.
- "Professional business attire" does NOT mean you should extract shirt, pants, jacket, tie and shoes. garments must be [].
- "Casual weekend outfit" does NOT mean you should extract t-shirt, jeans and sneakers. garments must be [].
- "Formal setting" does NOT mean "office". Set scene.label to null unless a specific place is named.
- Each garment's colour must come from the word directly modifying that garment in the text.
- Do NOT assign the same colour to multiple garments unless the query explicitly says so.

VALID YOLO CLASSES:
{yolo_classes}

OUTPUT SCHEMA (JSON only):
{{
  "global_refined_query": "Complete sentence preserving ALL query elements including actions, spatial relationships, garments, colours, scene and style. Keep the full sentence structure.",
  "garments": [
    {{
      "label": "exact YOLO class name or null if no garment mentioned",
      "color": "colour directly modifying this garment or null",
      "description": "exact phrase for regional garment CLIP matching",
      "explicit": true
    }}
  ],
  "scene": {{
    "label": "specific place name ONLY if mentioned, else null",
    "description": "scene description text",
    "formality": "formal" | "casual" | "sporty" | null
  }},
  "style_terms": ["style or context words from the query"]
}}

FEW-SHOT EXAMPLES:

Query: "A person in a bright yellow raincoat."
Correct: {{"global_refined_query": "A person wearing a bright yellow raincoat", "garments": [{{"label": "coat", "color": "yellow", "description": "bright yellow raincoat", "explicit": true}}], "scene": {{"label": null, "description": null, "formality": null}}, "style_terms": []}}

Query: "Professional business attire inside a modern office."
Correct: {{"global_refined_query": "Professional business attire in a modern office", "garments": [], "scene": {{"label": "office", "description": "modern office", "formality": "formal"}}, "style_terms": ["professional", "business attire", "modern"]}}

Query: "Someone wearing a blue shirt sitting on a park bench."
Correct: {{"global_refined_query": "Someone wearing a blue shirt sitting on a park bench", "garments": [{{"label": "shirt, blouse", "color": "blue", "description": "blue shirt", "explicit": true}}], "scene": {{"label": "park", "description": "park bench", "formality": null}}, "style_terms": []}}

Query: "Casual weekend outfit for a city walk."
Correct: {{"global_refined_query": "Casual weekend outfit for a city walk", "garments": [], "scene": {{"label": "street", "description": "city walk", "formality": "casual"}}, "style_terms": ["casual", "weekend outfit", "city walk"]}}

Query: "A red tie and a white shirt in a formal setting."
Correct: {{"global_refined_query": "A red tie and a white shirt in a formal setting", "garments": [{{"label": "tie", "color": "red", "description": "red tie", "explicit": true}}, {{"label": "shirt, blouse", "color": "white", "description": "white shirt", "explicit": true}}], "scene": {{"label": null, "description": "formal setting", "formality": "formal"}}, "style_terms": []}}"""


def _find_nearest_color(text, garment_start, garment_end):
    """Find the colour keyword closest to a garment mention in the text."""
    best_color = None
    best_dist = float("inf")
    for color in COLOR_KEYWORDS:
        for m in re.finditer(r"\b" + re.escape(color) + r"\b", text):
            color_center = (m.start() + m.end()) / 2
            garment_center = (garment_start + garment_end) / 2
            dist = abs(color_center - garment_center)
            if dist < best_dist:
                best_dist = dist
                best_color = color
    return best_color


def _rule_based_parse(text):
    """Deterministic rule-based fallback when LLM API is unreachable or fails."""
    text_lower = text.lower()
    garments = []

    parts = re.split(r"\band\b|,|\bwith\b", text_lower)

    for part in parts:
        part = part.strip()
        found_alias = None
        found_yolo = None
        garment_match = None

        for alias, yolo_cls in COMMON_GARMENT_ALIASES.items():
            m = re.search(r"\b" + re.escape(alias) + r"\b", part)
            if m:
                found_alias = alias
                found_yolo = yolo_cls
                garment_match = m
                break

        if not found_alias:
            continue

        color = None
        if garment_match:
            color = _find_nearest_color(
                part, garment_match.start(), garment_match.end()
            )

        garments.append({
            "label": found_yolo,
            "color": color or "unknown",
            "description": f"{color} {found_alias}".strip() if color else found_alias,
            "explicit": True,
        })

    found_scene = "general"
    for scene in SCENE_KEYWORDS:
        if re.search(r"\b" + re.escape(scene) + r"\b", text_lower):
            found_scene = scene
            break

    formality = None
    if any(w in text_lower for w in ["formal", "business", "professional", "office", "attire"]):
        formality = "formal"
    elif any(w in text_lower for w in ["sporty", "gym", "workout", "athletic", "running"]):
        formality = "sporty"
    elif any(w in text_lower for w in ["casual", "weekend", "everyday", "relaxed"]):
        formality = "casual"

    style_terms = []
    for term in ["professional", "business attire", "casual", "weekend", "formal", "modern"]:
        if term in text_lower:
            style_terms.append(term)

    return {
        "global_refined_query": text,
        "garments": garments,
        "scene": {
            "label": found_scene if found_scene != "general" else None,
            "description": f"{found_scene} environment" if found_scene != "general" else None,
            "formality": formality,
        },
        "style_terms": style_terms,
        "parser_source": "rule_based_fallback",
        "fallback_reason": None,
    }


def parse_query(text):
    """
    Parse a natural language query into structured components using OpenAI SDK + HF Router.
    Falls back cleanly to rule-based regex parsing if API is offline or credits are depleted.

    Output Schema:
    {
      "global_refined_query": str,
      "garments": [{"label": str, "color": str, "description": str, "explicit": bool}],
      "scene": {"label": str|null, "description": str|null, "formality": str|null},
      "style_terms": [str],
      "parser_source": "llm" | "rule_based_fallback",
      "fallback_reason": str|null
    }
    """
    if OpenAI is None:
        logger.warning("openai package not installed; using rule-based query parser")
        result = _rule_based_parse(text)
        result["fallback_reason"] = "openai package not installed"
        return result

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not api_key:
        logger.warning("HF_TOKEN not found in environment; using rule-based query parser")
        result = _rule_based_parse(text)
        result["fallback_reason"] = "HF_TOKEN not in environment"
        return result

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(yolo_classes=json.dumps(YOLO_FASHION_CLASSES))},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
            timeout=10.0
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        if "global_refined_query" not in parsed:
            parsed["global_refined_query"] = text
        if "garments" not in parsed or not isinstance(parsed["garments"], list):
            parsed["garments"] = []
        if "scene" not in parsed or not isinstance(parsed["scene"], dict):
            parsed["scene"] = {"label": None, "description": None, "formality": None}
        if "style_terms" not in parsed or not isinstance(parsed["style_terms"], list):
            parsed["style_terms"] = []

        for g in parsed["garments"]:
            if "explicit" not in g:
                g["explicit"] = True

        parsed["parser_source"] = "llm"
        parsed["fallback_reason"] = None
        return parsed

    except Exception as e:
        logger.warning("LLM query parser failed (%s); falling back to rule-based parser", e)
        result = _rule_based_parse(text)
        result["fallback_reason"] = str(e)
        return result
