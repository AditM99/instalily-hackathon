import re
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def preprocess_image(image):
    """Enhance receipt image for better OCR results."""
    # Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)

    # Resize if too small
    w, h = image.size
    if w < 800:
        ratio = 800 / w
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    return image


_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def extract_text(image):
    """
    Extract text from a receipt image using EasyOCR.
    Returns list of dicts with text, confidence, and bbox.
    """
    reader = _get_reader()

    img_array = np.array(image.convert("RGB"))
    results = reader.readtext(img_array)

    extracted = []
    for (bbox, text, confidence) in results:
        extracted.append({
            "text": text.strip(),
            "confidence": float(confidence),
            "bbox": bbox,
        })

    return extracted


def _is_price(text):
    """Check if a text block looks like a standalone price."""
    return bool(re.match(r"^\$?\s*\d+\.\d{2}\s*$", text.strip()))


def _extract_price(text):
    """Extract a price value from text. Returns (price, remaining_text) or (None, text)."""
    # Price at end of line: "Milk 1 Gal 3.99" or "Milk 1 Gal $3.99"
    match = re.search(r"\$?\s*(\d+\.\d{2})\s*$", text)
    if match:
        price = float(match.group(1))
        remaining = text[:match.start()].strip()
        remaining = re.sub(r"[\$]", "", remaining).strip()
        return price, remaining

    # Price anywhere in text: "3.99 Milk" or "$3.99"
    match = re.search(r"\$?\s*(\d+\.\d{2})", text)
    if match:
        price = float(match.group(1))
        remaining = text[:match.start()] + text[match.end():]
        remaining = re.sub(r"[\$]", "", remaining).strip()
        return price, remaining

    return None, text


def parse_receipt(ocr_results):
    """
    Parse structured data from OCR results.
    Handles EasyOCR's tendency to split item names and prices into separate blocks.
    """
    if not ocr_results:
        return {
            "store_name": "Unknown",
            "date": None,
            "items": [],
            "total": None,
            "tax": None,
            "low_confidence": [],
            "raw_text": "",
        }

    texts = [r["text"] for r in ocr_results]
    raw_text = "\n".join(texts)

    # Store name: typically the first non-empty, non-price line
    store_name = "Unknown"
    store_name_idx = -1
    for idx, t in enumerate(texts):
        if t and not _is_price(t) and len(t) > 1:
            store_name = t
            store_name_idx = idx
            break

    # Header lines (store name + next few lines before items start) are usually
    # address, phone, date — skip them. We mark the header zone as everything
    # before the first line that contains a price.
    header_end = 0
    for idx, t in enumerate(texts):
        if _is_price(t) or re.search(r"\d+\.\d{2}", t):
            header_end = idx
            break
    # Ensure header covers at least the store name
    header_end = max(header_end, store_name_idx + 1)

    # Date: look for date patterns and normalize to YYYY-MM-DD
    receipt_date = None
    date_patterns = [
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y"),
        (r"\d{1,2}/\d{1,2}/\d{2}", "%m/%d/%y"),
        (r"\d{1,2}-\d{1,2}-\d{4}", "%m-%d-%Y"),
        (r"\d{1,2}-\d{1,2}-\d{2}", "%m-%d-%y"),
    ]
    for text in texts:
        for pattern, fmt in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    from datetime import datetime
                    parsed_date = datetime.strptime(match.group(), fmt)
                    receipt_date = parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    receipt_date = match.group()
                break
        if receipt_date:
            break

    # Skip patterns — aggressively filter non-item receipt text
    skip_pattern = re.compile(
        r"\b(subtotal|sub\s*total|change|cash|card|visa|mastercard|debit|credit|"
        r"thank|welcome|receipt|store|address|phone|tel|fax|www|http|"
        r"refund|discount|saving|member|loyalty|reward|points|"
        r"approved|transaction|trans|auth|account|acct|ref|"
        r"cardholder|signature|retain|copy|duplicate|"
        r"chip|verified|contactless|terminal|merchant|"
        r"street|avenue|boulevard|road|drive|lane|suite|ste|"
        r"city|state|zip|county|country|"
        r"served\s*by|cashier|register|check\s*#|"
        r"items?\s*sold|item\s*count|"
        r"payment|paid|tendered|amount\s*tendered|"
        r"change\s*due|balance\s*due|"
        r"open|close|#\d+|"
        # Scanner / device / operational noise
        r"scan|scanner|scanned|barcode|qr\s*code|"
        r"powered\s*by|generated|printed|"
        # Names / staff
        r"manager|supervisor|employee|staff|clerk|operator|associate|"
        r"served|server|waiter|waitress|host|"
        # Greetings / footer messages
        r"have\s*a\s*(nice|great|good)|see\s*you|come\s*again|visit\s*us|"
        r"thank\s*you|thanks|goodbye|good\s*bye|"
        # Location / address fragments
        r"plaza|mall|center|centre|floor|level|unit|"
        r"st\b|ave\b|blvd\b|rd\b|dr\b|ln\b|ct\b|"
        r"\d+\s+(st|ave|blvd|rd|dr|ln|ct|street|avenue|road|drive)\b|"
        # Receipt identifiers
        r"order\s*#|invoice|receipt\s*#|ticket|"
        r"table\s*#|table\s*\d|seq|sequence|batch|"
        # Quantity prefixes (not items themselves)
        r"^qty|^quantity)\b",
        re.IGNORECASE,
    )
    total_pattern = re.compile(
        r"\b(total|amount\s*due|grand\s*total|balance|net\s*total)\b",
        re.IGNORECASE,
    )
    tax_pattern = re.compile(r"\b(tax|vat|gst|hst)\b", re.IGNORECASE)

    # Patterns for non-item text that should be skipped entirely
    _phone_pattern = re.compile(r"\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}")
    _zipcode_pattern = re.compile(r"\b\d{5}(-\d{4})?\b")
    _time_pattern = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm|AM|PM)?\b")
    _allcaps_header = re.compile(r"^[A-Z\s\#\-\.\,]{4,}$")  # e.g. "STORE #1234"
    _transaction_id = re.compile(r"^[A-Za-z]*#?\s*\d{4,}$")  # e.g. "TXN# 123456"
    _barcode_pattern = re.compile(r"^\d{8,}$")  # long digit strings (barcodes/UPCs)
    _address_pattern = re.compile(r"^\d+\s+\w+\s+(st|ave|blvd|rd|dr|ln|ct|street|avenue|road|drive|lane|way|place|pl)\b", re.IGNORECASE)
    _city_state_zip = re.compile(r"^[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}", re.IGNORECASE)  # "Springfield, IL 62701"
    _url_pattern = re.compile(r"(www\.|\.com|\.net|\.org|https?://)", re.IGNORECASE)

    items = []
    total = None
    tax = None

    # Pre-merge: combine adjacent non-price text blocks on the same line
    # EasyOCR often splits "Bananas 2lb" into ["Bananas", "2lb"] or
    # "Whole Milk 1Gal" into ["Whole", "Milk", "1Gal"]
    merged_results = []
    for entry in ocr_results:
        text = entry["text"].strip()
        if not text:
            continue
        if (merged_results
                and not _is_price(text)
                and not _is_price(merged_results[-1]["text"])
                and not re.search(r"\d+\.\d{2}", merged_results[-1]["text"])
                and not re.search(r"\d+\.\d{2}", text)
                and abs(entry["bbox"][0][1] - merged_results[-1]["bbox"][0][1]) < 15):
            # Same line, neither has a price — merge
            merged_results[-1]["text"] += " " + text
            merged_results[-1]["confidence"] = min(merged_results[-1]["confidence"], entry["confidence"])
        else:
            merged_results.append(dict(entry))  # copy to avoid mutating original
    ocr_results = merged_results

    # Recalculate header_end and store_name after merge
    texts = [r["text"] for r in ocr_results]
    store_name = "Unknown"
    store_name_idx = -1
    for idx, t in enumerate(texts):
        if t and not _is_price(t) and len(t) > 1:
            store_name = t
            store_name_idx = idx
            break

    # Header ends just before the first item+price pair.
    # Find the first standalone price — the item name is the block before it.
    header_end = 0
    for idx, t in enumerate(texts):
        if _is_price(t) or re.search(r"\d+\.\d{2}", t):
            # If previous block is a non-price text (i.e. item name), header ends before it
            if idx > 0 and not _is_price(texts[idx - 1]) and not skip_pattern.search(texts[idx - 1]):
                header_end = idx - 1
            else:
                header_end = idx
            break
    header_end = max(header_end, store_name_idx + 1)

    # Strategy: iterate through OCR results and try to pair text blocks
    # EasyOCR may give "Milk 1 Gal" and "3.99" as separate entries
    # or "Milk 1 Gal 3.99" as one entry
    i = 0
    while i < len(ocr_results):
        entry = ocr_results[i]
        text = entry["text"].strip()
        confidence = entry["confidence"]

        # Skip empty
        if not text:
            i += 1
            continue

        # Skip header zone (store name, address, phone, etc.)
        # But still check for total/tax in the header in case of weird formatting
        if i < header_end and not total_pattern.search(text) and not tax_pattern.search(text):
            i += 1
            continue

        # Check for total line
        if total_pattern.search(text):
            price, _ = _extract_price(text)
            if price:
                total = price
            elif i + 1 < len(ocr_results):
                # Price might be the next block
                next_text = ocr_results[i + 1]["text"].strip()
                price, _ = _extract_price(next_text)
                if price:
                    total = price
                    i += 1
            i += 1
            continue

        # Check for tax line
        if tax_pattern.search(text):
            price, _ = _extract_price(text)
            if price:
                tax = price
            elif i + 1 < len(ocr_results):
                next_text = ocr_results[i + 1]["text"].strip()
                price, _ = _extract_price(next_text)
                if price:
                    tax = price
                    i += 1
            i += 1
            continue

        # Skip non-item lines
        if skip_pattern.search(text):
            i += 1
            continue

        # Skip date lines
        is_date = False
        for pattern, _fmt in date_patterns:
            if re.search(pattern, text):
                is_date = True
                break
        if is_date:
            i += 1
            continue

        # Skip phone numbers, zip codes, timestamps, transaction IDs
        if (_phone_pattern.search(text) or _time_pattern.search(text)
                or _transaction_id.match(text)):
            i += 1
            continue

        # Skip barcodes (long digit-only strings), addresses, city/state/zip, URLs
        if (_barcode_pattern.match(text) or _address_pattern.match(text)
                or _city_state_zip.match(text) or _url_pattern.search(text)):
            i += 1
            continue

        # Skip lines that are just a zip code or look like an address fragment
        stripped_noprice = re.sub(r"\$?\d+\.\d{2}", "", text).strip()
        if _zipcode_pattern.fullmatch(stripped_noprice):
            i += 1
            continue

        # Skip ALL-CAPS header lines that aren't items (e.g. store name, address)
        if _allcaps_header.match(text) and not re.search(r"\d+\.\d{2}", text):
            i += 1
            continue

        # Skip lines that are mostly digits with no price format (IDs, codes)
        digits_only = re.sub(r"[^0-9]", "", text)
        if len(digits_only) > 0 and len(digits_only) / len(text) > 0.7 and not re.search(r"\d+\.\d{2}", text):
            i += 1
            continue

        # Try to extract price from this block
        price, item_name = _extract_price(text)

        if price and item_name:
            # Price and name in same block: "Milk 1 Gal 3.99"
            items.append({
                "name": item_name,
                "price": price,
                "confidence": confidence,
            })
            i += 1
            continue

        if price and not item_name:
            # Standalone price - might belong to previous text block
            # Check if previous item has no price (was just a name)
            if items and items[-1].get("_needs_price"):
                items[-1]["price"] = price
                del items[-1]["_needs_price"]
            i += 1
            continue

        if not price and item_name:
            # Text with no price - check if next block is a standalone price
            if i + 1 < len(ocr_results):
                next_text = ocr_results[i + 1]["text"].strip()
                next_price, next_remaining = _extract_price(next_text)
                if next_price and (not next_remaining or len(next_remaining) <= 2):
                    # Next block is a price for this item
                    items.append({
                        "name": text,
                        "price": next_price,
                        "confidence": confidence,
                    })
                    i += 2
                    continue

            # No price found - store as item needing a price
            # Only if it looks like an actual item name
            if (len(text) > 2
                    and not _allcaps_header.match(text)
                    and not _phone_pattern.search(text)
                    and not _zipcode_pattern.fullmatch(text)
                    and not _time_pattern.search(text)):
                items.append({
                    "name": text,
                    "price": 0,
                    "confidence": confidence,
                    "_needs_price": True,
                })
            i += 1
            continue

        i += 1

    # Clean up items that never got a price
    items = [item for item in items if item.get("price", 0) > 0 and "_needs_price" not in item]

    # Remove store name from items if it got picked up
    items = [item for item in items if item["name"].lower() != store_name.lower()]

    # Identify low-confidence items for clarification agent
    low_confidence = []
    CONFIDENCE_THRESHOLD = 0.7
    for entry in ocr_results:
        if entry["confidence"] < CONFIDENCE_THRESHOLD:
            low_confidence.append({
                "text": entry["text"],
                "confidence": entry["confidence"],
                "bbox": entry["bbox"],
            })

    return {
        "store_name": store_name,
        "date": receipt_date,
        "items": items,
        "total": total,
        "tax": tax,
        "low_confidence": low_confidence,
        "raw_text": raw_text,
    }


def crop_region(image, bbox):
    """Crop a region from the image based on bounding box for clarification."""
    points = np.array(bbox)
    x_min = int(points[:, 0].min())
    y_min = int(points[:, 1].min())
    x_max = int(points[:, 0].max())
    y_max = int(points[:, 1].max())

    # Add some padding
    pad = 10
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(image.width, x_max + pad)
    y_max = min(image.height, y_max + pad)

    return image.crop((x_min, y_min, x_max, y_max))


def process_receipt_image(image):
    """Full pipeline: preprocess -> OCR -> parse."""
    preprocessed = preprocess_image(image)
    ocr_results = extract_text(preprocessed)
    parsed = parse_receipt(ocr_results)
    return parsed, preprocessed
