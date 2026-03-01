"""Inference wrapper for the fine-tuned receipt item categorizer."""

import os
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "categorizer")

CATEGORIES = [
    "groceries", "dining", "transport", "entertainment",
    "health", "clothing", "utilities", "other",
]

# Keyword fallback in case model isn't trained yet
KEYWORD_MAP = {
    "groceries": ["milk", "bread", "egg", "cheese", "chicken", "beef", "rice", "pasta",
                   "fruit", "vegetable", "apple", "banana", "lettuce", "tomato", "onion",
                   "potato", "butter", "yogurt", "cereal", "juice", "water", "frozen",
                   "canned", "flour", "sugar", "oil", "salt", "pepper", "cream"],
    "dining": ["coffee", "latte", "cappuccino", "espresso", "burger", "sandwich", "pizza",
               "sushi", "taco", "burrito", "fries", "wings", "salad", "steak", "beer",
               "wine", "cocktail", "restaurant", "cafe", "diner", "bar", "grill",
               "smoothie", "frappe", "mocha", "pancake", "waffle"],
    "transport": ["gas", "fuel", "diesel", "uber", "lyft", "taxi", "bus", "metro",
                  "parking", "toll", "car wash", "oil change", "tire", "brake"],
    "entertainment": ["movie", "cinema", "netflix", "spotify", "ticket", "game",
                      "book", "magazine", "concert", "theater", "museum", "streaming"],
    "health": ["prescription", "vitamin", "medicine", "pharmacy", "doctor", "dental",
               "gym", "yoga", "protein", "supplement", "bandaid", "tylenol", "aspirin"],
    "clothing": ["shirt", "jeans", "dress", "jacket", "shoes", "boots", "socks",
                 "sweater", "coat", "hat", "belt", "sneaker", "sandal"],
    "utilities": ["electric", "water bill", "internet", "phone bill", "cable",
                  "detergent", "soap", "cleaner", "paper towel", "toilet paper",
                  "trash bag", "light bulb", "battery", "charger"],
}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return True

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return False

    _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()
    return True


def _check_corrections(item_name):
    """Check if the agent has previously corrected this item's category."""
    try:
        import db as _db
        corrections = _db.get_category_corrections(item_name)
        if corrections and corrections[0].get("new_category"):
            return corrections[0]["new_category"], 0.95
    except Exception:
        pass
    return None, None


def categorize_item(item_name):
    """
    Categorize a receipt item.
    Returns (category, confidence).
    Priority: agent corrections > fine-tuned model > keyword fallback.
    """
    # 1. Check if the agent has corrected this item before (learning from feedback)
    corrected_cat, corrected_conf = _check_corrections(item_name)
    if corrected_cat:
        return corrected_cat, corrected_conf

    # 2. Use fine-tuned model if available
    if _load_model():
        return _categorize_with_model(item_name)

    # 3. Keyword fallback
    return _categorize_with_keywords(item_name)


def _categorize_with_model(item_name):
    """Categorize using the fine-tuned DistilBERT model."""
    inputs = _tokenizer(
        item_name,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)

    category = CATEGORIES[predicted.item()]
    return category, round(confidence.item(), 3)


def _categorize_with_keywords(item_name):
    """Fallback: keyword-based categorization."""
    name_lower = item_name.lower()

    for category, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category, 0.8  # Fixed confidence for keyword match

    return "other", 0.5


def categorize_items(items):
    """Categorize a list of items. Each item is a dict with 'name' and 'price'."""
    results = []
    for item in items:
        category, confidence = categorize_item(item["name"])
        results.append({
            **item,
            "category": category,
            "original_category": category,
            "model_confidence": confidence,
        })
    return results
