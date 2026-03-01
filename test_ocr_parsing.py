"""Quick test for the receipt parser logic — no OCR needed, just simulated OCR output."""

from ocr import parse_receipt

# Simulate what EasyOCR would return for a real receipt with addresses, timestamps, etc.
fake_ocr_results = [
    {"text": "WALMART", "confidence": 0.95, "bbox": [[0,0],[100,0],[100,20],[0,20]]},
    {"text": "SUPERCENTER", "confidence": 0.90, "bbox": [[0,20],[100,20],[100,40],[0,40]]},
    {"text": "1234 Main St", "confidence": 0.88, "bbox": [[0,40],[100,40],[100,60],[0,60]]},
    {"text": "Springfield, IL 62704", "confidence": 0.85, "bbox": [[0,60],[100,60],[100,80],[0,80]]},
    {"text": "(217) 555-1234", "confidence": 0.92, "bbox": [[0,80],[100,80],[100,100],[0,100]]},
    {"text": "01/15/2024", "confidence": 0.95, "bbox": [[0,100],[100,100],[100,120],[0,120]]},
    {"text": "10:34 AM", "confidence": 0.90, "bbox": [[0,120],[100,120],[100,140],[0,140]]},
    {"text": "Cashier: Jane", "confidence": 0.88, "bbox": [[0,140],[100,140],[100,160],[0,160]]},
    {"text": "Register #3", "confidence": 0.91, "bbox": [[0,160],[100,160],[100,180],[0,180]]},
    {"text": "Milk 1 Gal 3.99", "confidence": 0.93, "bbox": [[0,200],[200,200],[200,220],[0,220]]},
    {"text": "Bread Wheat", "confidence": 0.91, "bbox": [[0,220],[200,220],[200,240],[0,240]]},
    {"text": "2.49", "confidence": 0.94, "bbox": [[200,220],[250,220],[250,240],[200,240]]},
    {"text": "Eggs 12ct 3.29", "confidence": 0.92, "bbox": [[0,240],[200,240],[200,260],[0,260]]},
    {"text": "Chicken Breast", "confidence": 0.89, "bbox": [[0,260],[200,260],[200,280],[0,280]]},
    {"text": "7.49", "confidence": 0.93, "bbox": [[200,260],[250,260],[250,280],[200,280]]},
    {"text": "Tax 0.78", "confidence": 0.95, "bbox": [[0,300],[200,300],[200,320],[0,320]]},
    {"text": "Total 18.04", "confidence": 0.96, "bbox": [[0,320],[200,320],[200,340],[0,340]]},
    {"text": "VISA **** 1234", "confidence": 0.88, "bbox": [[0,340],[200,340],[200,360],[0,360]]},
    {"text": "Approved", "confidence": 0.92, "bbox": [[0,360],[200,360],[200,380],[0,380]]},
    {"text": "Transaction #847291", "confidence": 0.90, "bbox": [[0,380],[200,380],[200,400],[0,400]]},
    {"text": "THANK YOU FOR SHOPPING", "confidence": 0.91, "bbox": [[0,400],[200,400],[200,420],[0,420]]},
]

parsed = parse_receipt(fake_ocr_results)

print("Store:", parsed["store_name"])
print("Date:", parsed["date"])
print("Tax:", parsed["tax"])
print("Total:", parsed["total"])
print()
print("Items found:")
for item in parsed["items"]:
    print(f"  {item['name']} — ${item['price']:.2f}")

print()
# Verify correctness
expected_items = {"Milk 1 Gal", "Bread Wheat", "Eggs 12ct", "Chicken Breast"}
actual_items = {item["name"] for item in parsed["items"]}

# Check nothing unexpected leaked through
unexpected = actual_items - expected_items
if unexpected:
    print(f"PROBLEM: Unexpected items detected: {unexpected}")
else:
    print("OK: Only real items were detected (no addresses, timestamps, etc.)")

missing = expected_items - actual_items
if missing:
    print(f"WARNING: Missing expected items: {missing}")
else:
    print("OK: All expected items were found")

assert parsed["total"] == 18.04, f"Total wrong: {parsed['total']}"
assert parsed["tax"] == 0.78, f"Tax wrong: {parsed['tax']}"
print("OK: Total and tax correct")
