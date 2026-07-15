import os
import re
import json

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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


def _rule_based_parse(text):
    """Deterministic rule-based fallback when LLM API is unreachable or fails."""
    text_lower = text.lower()
    garments = []

    # Find garment terms and their surrounding color words
    for alias, yolo_cls in COMMON_GARMENT_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            # Check for colors nearby
            found_color = "unknown"
            for color in COLOR_KEYWORDS:
                if re.search(r"\b" + re.escape(color) + r"\b", text_lower):
                    found_color = color
                    break
            garments.append({
                "label": yolo_cls,
                "color": found_color if found_color != "unknown" else "neutral",
                "description": f"{found_color} {alias}".strip() if found_color != "unknown" else alias
            })

    # Find scene terms
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

    return {
        "global_refined_query": text,
        "garments": garments,
        "scene": {
            "label": found_scene,
            "description": f"{found_scene} environment" if found_scene != "general" else text,
            "formality": formality
        }
    }


def parse_query(text):
    """
    Parse a natural language query into structured components using OpenAI SDK + HF Router.
    Falls back cleanly to high-accuracy rule-based regex parsing if API is offline.

    Output Schema:
    {
      "global_refined_query": str,
      "garments": [{"label": str, "color": str, "description": str}],
      "scene": {"label": str, "description": str, "formality": "formal"|"casual"|"sporty"|null}
    }
    """
    if OpenAI is None:
        return _rule_based_parse(text)

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not api_key:
        return _rule_based_parse(text)
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )
        system_prompt = f"""You are a fashion query parsing specialist.
Analyze the user's natural language fashion search query and decompose it into structured JSON.

You must map garment mentions to the closest matching Fashionpedia YOLO classes from this list:
{json.dumps(YOLO_FASHION_CLASSES)}

Return ONLY a valid JSON object matching this exact schema:
{{
  "global_refined_query": "Overall description of scene & style for global CLIP matching",
  "garments": [
    {{
      "label": "exact YOLO class name from the provided list",
      "color": "normalized color name (e.g. red, blue, yellow, black, white, gray, brown, green, pink, purple, orange)",
      "description": "exact phrase or refined text description for regional garment CLIP matching (e.g. 'bright yellow raincoat')"
    }}
  ],
  "scene": {{
    "label": "scene label (e.g. office, street, park, beach, home, restaurant, cafe)",
    "description": "scene description text",
    "formality": "formal" | "casual" | "sporty" | null
  }}
}}"""

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
            timeout=5.0
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Ensure schema keys exist
        if "global_refined_query" not in parsed:
            parsed["global_refined_query"] = text
        if "garments" not in parsed or not isinstance(parsed["garments"], list):
            parsed["garments"] = []
        if "scene" not in parsed or not isinstance(parsed["scene"], dict):
            parsed["scene"] = {"label": "general", "description": text, "formality": None}

        return parsed
    except Exception as e:
        # Fallback cleanly when API call fails
        return _rule_based_parse(text)
