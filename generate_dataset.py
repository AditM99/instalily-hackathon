"""Generate synthetic training data for receipt item categorization."""

import json
import random
import os

CATEGORIES = {
    "groceries": [
        "Milk", "Bread", "Eggs", "Butter", "Cheese", "Yogurt", "Chicken Breast",
        "Ground Beef", "Salmon", "Rice", "Pasta", "Tomato Sauce", "Olive Oil",
        "Salt", "Pepper", "Sugar", "Flour", "Cereal", "Oatmeal", "Bananas",
        "Apples", "Oranges", "Strawberries", "Blueberries", "Grapes", "Lettuce",
        "Tomatoes", "Onions", "Potatoes", "Carrots", "Broccoli", "Spinach",
        "Frozen Pizza", "Ice Cream", "Juice", "Water Bottles", "Canned Beans",
        "Peanut Butter", "Jam", "Honey", "Chips", "Crackers", "Cookies",
        "Cake Mix", "Baking Soda", "Vanilla Extract", "Coconut Milk",
        "Almond Milk", "Soy Sauce", "Ketchup", "Mustard", "Mayo",
        "Salad Dressing", "Tortillas", "Hot Dogs", "Bacon", "Sausage",
        "Deli Turkey", "Ham", "Cream Cheese", "Sour Cream", "Cottage Cheese",
    ],
    "dining": [
        "Latte", "Cappuccino", "Espresso", "Coffee", "Americano", "Mocha",
        "Frappe", "Iced Tea", "Smoothie", "Burger", "Cheeseburger",
        "French Fries", "Chicken Sandwich", "Club Sandwich", "BLT",
        "Caesar Salad", "Pizza Slice", "Pasta Bowl", "Sushi Roll",
        "Pad Thai", "Fried Rice", "Burrito", "Tacos", "Nachos",
        "Wings", "Mozzarella Sticks", "Onion Rings", "Milkshake",
        "Pancakes", "Waffles", "Omelette", "Breakfast Combo", "Soup",
        "Steak Dinner", "Fish and Chips", "Grilled Chicken", "Salad Bowl",
        "Dessert", "Cheesecake", "Brownie", "Ice Cream Sundae",
        "Beer", "Wine Glass", "Cocktail", "Soda", "Sparkling Water",
    ],
    "transport": [
        "Gas", "Gasoline", "Diesel", "Fuel", "Premium Gas", "Regular Gas",
        "Uber Ride", "Lyft Ride", "Taxi Fare", "Bus Pass", "Metro Card",
        "Subway Ticket", "Train Ticket", "Parking Fee", "Parking Meter",
        "Toll", "Highway Toll", "Bridge Toll", "Car Wash", "Oil Change",
        "Tire Rotation", "Brake Pads", "Windshield Wipers", "Coolant",
        "Air Filter", "Spark Plugs", "Car Battery", "Motor Oil",
    ],
    "entertainment": [
        "Movie Ticket", "Cinema", "Netflix", "Spotify", "HBO Max",
        "Disney Plus", "Hulu", "Amazon Prime", "YouTube Premium",
        "Concert Ticket", "Theater Ticket", "Museum Entry", "Zoo Ticket",
        "Amusement Park", "Bowling", "Mini Golf", "Arcade Games",
        "Video Game", "Board Game", "Book", "Magazine", "Newspaper",
        "Streaming Service", "Music Download", "App Purchase",
        "Escape Room", "Karaoke", "Comedy Show", "Sports Ticket",
    ],
    "health": [
        "Prescription", "Medication", "Vitamins", "Supplements",
        "Ibuprofen", "Tylenol", "Aspirin", "Cough Syrup", "Allergy Meds",
        "Band-Aids", "First Aid Kit", "Thermometer", "Blood Pressure Monitor",
        "Dental Floss", "Toothpaste", "Mouthwash", "Sunscreen",
        "Hand Sanitizer", "Face Mask", "Eye Drops", "Contact Lens Solution",
        "Doctor Copay", "Lab Work", "X-Ray", "Physical Therapy",
        "Gym Membership", "Yoga Class", "Protein Powder", "Health Drink",
    ],
    "clothing": [
        "T-Shirt", "Jeans", "Dress Shirt", "Blouse", "Sweater", "Jacket",
        "Coat", "Hoodie", "Shorts", "Skirt", "Dress", "Suit",
        "Socks", "Underwear", "Bra", "Tie", "Belt", "Scarf", "Gloves",
        "Hat", "Cap", "Sneakers", "Boots", "Sandals", "Dress Shoes",
        "Running Shoes", "Slippers", "Sunglasses", "Watch", "Jewelry",
        "Handbag", "Backpack", "Wallet",
    ],
    "utilities": [
        "Electric Bill", "Gas Bill", "Water Bill", "Internet Bill",
        "Phone Bill", "Cable TV", "Trash Service", "Sewer Bill",
        "Phone Case", "Phone Charger", "USB Cable", "HDMI Cable",
        "Light Bulbs", "Batteries", "Extension Cord", "Power Strip",
        "Printer Ink", "Paper", "Envelopes", "Stamps", "Cleaning Supplies",
        "Dish Soap", "Laundry Detergent", "Paper Towels", "Toilet Paper",
        "Trash Bags", "Sponges", "Bleach", "Glass Cleaner",
    ],
    "other": [
        "Gift Card", "Greeting Card", "Wrapping Paper", "Flowers",
        "Pet Food", "Dog Treats", "Cat Litter", "Pet Toy", "Plant",
        "Gardening Soil", "Seeds", "Fertilizer", "Tool Set", "Nails",
        "Screws", "Paint", "Paintbrush", "Tape", "Glue", "Scissors",
        "Donation", "Tip", "Service Charge", "Delivery Fee",
        "Membership Fee", "Annual Fee", "Late Fee", "ATM Fee",
    ],
}


def augment_item(item_name):
    """Generate variations of an item name."""
    variations = [item_name]

    # Uppercase
    variations.append(item_name.upper())
    # Lowercase
    variations.append(item_name.lower())
    # Abbreviated (first 6 chars)
    if len(item_name) > 6:
        variations.append(item_name[:6].upper())
    # With quantity prefix
    qty = random.choice(["1x ", "2x ", "3x ", "QTY 1 ", ""])
    variations.append(qty + item_name)
    # With random typo
    if len(item_name) > 3:
        pos = random.randint(1, len(item_name) - 2)
        typo = item_name[:pos] + random.choice("aeiou") + item_name[pos + 1:]
        variations.append(typo)

    return variations


def generate_dataset(output_path="dataset.json"):
    """Generate the full synthetic dataset."""
    data = []

    for category, items in CATEGORIES.items():
        for item in items:
            for variation in augment_item(item):
                data.append({
                    "text": variation.strip(),
                    "label": category,
                })

    # Shuffle
    random.shuffle(data)

    # Save
    filepath = os.path.join(os.path.dirname(__file__), output_path)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data)} examples across {len(CATEGORIES)} categories")
    print(f"Saved to {filepath}")

    # Print distribution
    from collections import Counter
    dist = Counter(d["label"] for d in data)
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")

    return data


if __name__ == "__main__":
    generate_dataset()
