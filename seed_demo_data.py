"""Populate the database with 30 days of realistic fake receipt data for demo."""

import random
from datetime import datetime, timedelta
import db

STORES = {
    "Walmart": {"categories": ["groceries"], "avg_items": 8, "price_range": (1.5, 15)},
    "Target": {"categories": ["groceries", "clothing", "other"], "avg_items": 5, "price_range": (3, 40)},
    "Starbucks": {"categories": ["dining"], "avg_items": 2, "price_range": (3.5, 7)},
    "McDonald's": {"categories": ["dining"], "avg_items": 3, "price_range": (2, 10)},
    "Chipotle": {"categories": ["dining"], "avg_items": 2, "price_range": (8, 15)},
    "Shell Gas": {"categories": ["transport"], "avg_items": 1, "price_range": (35, 65)},
    "CVS Pharmacy": {"categories": ["health", "groceries"], "avg_items": 3, "price_range": (4, 25)},
    "Amazon": {"categories": ["entertainment", "other"], "avg_items": 2, "price_range": (10, 50)},
    "Costco": {"categories": ["groceries"], "avg_items": 10, "price_range": (5, 30)},
    "Olive Garden": {"categories": ["dining"], "avg_items": 3, "price_range": (10, 25)},
    "H&M": {"categories": ["clothing"], "avg_items": 2, "price_range": (15, 45)},
    "Best Buy": {"categories": ["utilities", "entertainment"], "avg_items": 1, "price_range": (20, 100)},
}

ITEMS_BY_CATEGORY = {
    "groceries": [
        ("Milk 1 Gal", 3.99), ("Bread White", 2.49), ("Eggs 12ct", 3.29),
        ("Chicken Breast", 7.99), ("Ground Beef", 6.49), ("Bananas", 1.29),
        ("Apples 3lb", 4.99), ("Rice 5lb", 5.49), ("Pasta Box", 1.79),
        ("Tomato Sauce", 2.29), ("Cheese Block", 4.49), ("Yogurt 4pk", 3.99),
        ("Orange Juice", 3.79), ("Cereal Box", 4.29), ("Frozen Pizza", 5.99),
        ("Lettuce Head", 1.99), ("Tomatoes", 2.99), ("Onions 3lb", 2.49),
        ("Potatoes 5lb", 4.99), ("Butter", 3.49),
    ],
    "dining": [
        ("Latte Grande", 5.25), ("Cappuccino", 4.75), ("Iced Coffee", 4.50),
        ("Espresso", 3.25), ("Mocha Frappe", 5.99), ("Burger Combo", 9.99),
        ("Chicken Sandwich", 8.49), ("French Fries", 3.99), ("Caesar Salad", 11.99),
        ("Pasta Bowl", 14.99), ("Pizza Slice", 3.50), ("Burrito Bowl", 12.75),
        ("Steak Dinner", 22.99), ("Fish Tacos", 13.49), ("Wings 10pc", 14.99),
        ("Pancake Stack", 8.99), ("Soda", 2.49), ("Beer Draft", 6.99),
    ],
    "transport": [
        ("Regular Gas", 45.00), ("Premium Gas", 55.00), ("Diesel", 50.00),
        ("Car Wash", 12.00), ("Parking Fee", 8.00), ("Oil Change", 39.99),
    ],
    "entertainment": [
        ("Movie Ticket", 14.99), ("Netflix Sub", 15.49), ("Spotify Sub", 10.99),
        ("Video Game", 39.99), ("Book", 12.99), ("Concert Ticket", 45.00),
    ],
    "health": [
        ("Vitamins", 12.99), ("Ibuprofen", 7.99), ("Bandaids", 4.49),
        ("Sunscreen", 9.99), ("Toothpaste", 3.99), ("Cough Syrup", 8.99),
        ("Allergy Meds", 11.99), ("Hand Sanitizer", 3.49),
    ],
    "clothing": [
        ("T-Shirt", 14.99), ("Jeans", 39.99), ("Sneakers", 59.99),
        ("Socks 6pk", 9.99), ("Hoodie", 29.99), ("Belt", 19.99),
    ],
    "utilities": [
        ("Phone Charger", 14.99), ("USB Cable", 9.99), ("Light Bulbs 4pk", 7.99),
        ("Batteries 8pk", 8.99), ("Trash Bags", 6.99), ("Paper Towels", 5.99),
        ("Laundry Detergent", 11.99), ("Dish Soap", 3.99),
    ],
    "other": [
        ("Gift Card", 25.00), ("Greeting Card", 4.99), ("Dog Food", 24.99),
        ("Cat Litter", 12.99), ("Plant Pot", 8.99), ("Tape Roll", 3.49),
    ],
}


def seed(days=30, receipts_per_day_range=(1, 3)):
    """Generate realistic demo data. Always generates within the current month."""
    # Reset DB
    db.init_db()

    today = datetime.now().date()
    # Start from the 1st of this month so all data falls in the current calendar month
    month_start = today.replace(day=1)
    # Generate from month start up to yesterday (so we have a full month of data)
    actual_days = (today - month_start).days
    if actual_days < 5:
        # If early in the month, include some days from last month too
        actual_days = days
        start_date = today - timedelta(days=actual_days)
    else:
        start_date = month_start
        actual_days = actual_days  # days within this month

    receipt_count = 0

    for day_offset in range(actual_days, 0, -1):
        receipt_date = today - timedelta(days=day_offset)
        num_receipts = random.randint(*receipts_per_day_range)

        for _ in range(num_receipts):
            store_name = random.choice(list(STORES.keys()))
            store_info = STORES[store_name]

            # Pick items from relevant categories
            items = []
            num_items = random.randint(1, store_info["avg_items"])
            for _ in range(num_items):
                cat = random.choice(store_info["categories"])
                if cat in ITEMS_BY_CATEGORY:
                    item_name, base_price = random.choice(ITEMS_BY_CATEGORY[cat])
                    # Add some price variation
                    price = round(base_price * random.uniform(0.85, 1.15), 2)
                    items.append({
                        "name": item_name,
                        "price": price,
                        "category": cat,
                    })

            if not items:
                continue

            total = sum(i["price"] for i in items)
            tax = round(total * 0.08, 2)

            receipt_id = db.insert_receipt(
                store_name=store_name,
                receipt_date=str(receipt_date),
                total=round(total + tax, 2),
                tax=tax,
                image_path=None,
                raw_ocr_text=f"Receipt from {store_name}",
            )

            for item in items:
                db.insert_line_item(
                    receipt_id=receipt_id,
                    item_name=item["name"],
                    price=item["price"],
                    category=item["category"],
                    confidence=random.uniform(0.85, 0.99),
                )

            receipt_count += 1

    # Add some agent memories for context
    db.insert_agent_memory(
        "User's dining spending has been increasing over the past 2 weeks",
        "Warned about dining budget approaching limit",
    )
    db.insert_agent_memory(
        "User frequently visits Starbucks (avg 4x/week)",
        "Suggested reducing coffee shop visits to save $80/month",
    )

    print(f"Seeded {receipt_count} receipts over {days} days")


if __name__ == "__main__":
    seed()
