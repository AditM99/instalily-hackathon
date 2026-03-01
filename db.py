import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "receipt_agent.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT,
            receipt_date DATE,
            total REAL,
            tax REAL,
            image_path TEXT,
            raw_ocr_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS line_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER REFERENCES receipts(id),
            item_name TEXT,
            price REAL,
            category TEXT,
            confidence REAL
        );

        CREATE TABLE IF NOT EXISTS budgets (
            category TEXT PRIMARY KEY,
            monthly_limit REAL,
            user_override INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation TEXT,
            action_taken TEXT,
            receipt_id INTEGER REFERENCES receipts(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER REFERENCES receipts(id),
            line_item_id INTEGER REFERENCES line_items(id),
            reason TEXT,
            resolved INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            target_amount REAL,
            current_saved REAL DEFAULT 0,
            deadline DATE,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            detail TEXT,
            receipt_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS category_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            old_category TEXT,
            new_category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Add user_override column to budgets if it doesn't exist (migration)
    try:
        cursor.execute("SELECT user_override FROM budgets LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE budgets ADD COLUMN user_override INTEGER DEFAULT 0")

    # Pre-populate budgets with defaults if empty
    cursor.execute("SELECT COUNT(*) FROM budgets")
    if cursor.fetchone()[0] == 0:
        defaults = [
            ("groceries", 500),
            ("dining", 300),
            ("transport", 200),
            ("entertainment", 150),
            ("health", 100),
            ("clothing", 150),
            ("utilities", 200),
            ("other", 100),
        ]
        cursor.executemany(
            "INSERT INTO budgets (category, monthly_limit) VALUES (?, ?)", defaults
        )

    conn.commit()
    conn.close()


# --- CRUD Operations ---

def insert_receipt(store_name, receipt_date, total, tax, image_path, raw_ocr_text):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO receipts (store_name, receipt_date, total, tax, image_path, raw_ocr_text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (store_name, receipt_date, total, tax, image_path, raw_ocr_text),
    )
    receipt_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return receipt_id


def insert_line_item(receipt_id, item_name, price, category, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO line_items (receipt_id, item_name, price, category, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (receipt_id, item_name, price, category, confidence),
    )
    conn.commit()
    conn.close()


def insert_agent_memory(observation, action_taken, receipt_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO agent_memory (observation, action_taken, receipt_id)
           VALUES (?, ?, ?)""",
        (observation, action_taken, receipt_id),
    )
    conn.commit()
    conn.close()


# --- Query Functions (used by agent tools) ---

def query_spending(category=None, days=30):
    """Total spending by category. days=30 uses calendar month for consistency."""
    conn = get_connection()
    cursor = conn.cursor()

    # Use calendar month for "this month" (days=30) to match pie chart & budget sidebar.
    # Other values use rolling days.
    if days == 30:
        date_filter = "strftime('%Y-%m', r.receipt_date) = strftime('%Y-%m', 'now', 'localtime')"
        params = ()
    else:
        date_filter = "r.receipt_date >= date('now', ?, 'localtime')"
        params = (f"-{days} days",)

    if category:
        cursor.execute(
            f"""SELECT category, SUM(price) as total_spent, COUNT(*) as item_count
               FROM line_items li
               JOIN receipts r ON li.receipt_id = r.id
               WHERE li.category = ?
                 AND {date_filter}
               GROUP BY li.category""",
            (category, *params),
        )
    else:
        cursor.execute(
            f"""SELECT category, SUM(price) as total_spent, COUNT(*) as item_count
               FROM line_items li
               JOIN receipts r ON li.receipt_id = r.id
               WHERE {date_filter}
               GROUP BY li.category
               ORDER BY total_spent DESC""",
            params,
        )

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def check_budget(category):
    """Check remaining budget for a category this month."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT monthly_limit, user_override FROM budgets WHERE category = ?",
        (category,),
    )
    budget_row = cursor.fetchone()
    if not budget_row:
        conn.close()
        return {"category": category, "error": "No budget set"}

    monthly_limit = budget_row["monthly_limit"]
    user_override = budget_row["user_override"]

    cursor.execute(
        """SELECT COALESCE(SUM(price), 0) as spent
           FROM line_items li
           JOIN receipts r ON li.receipt_id = r.id
           WHERE li.category = ?
             AND strftime('%Y-%m', r.receipt_date) = strftime('%Y-%m', 'now', 'localtime')""",
        (category,),
    )
    spent = cursor.fetchone()["spent"]
    conn.close()

    remaining = monthly_limit - spent
    percent_used = (spent / monthly_limit * 100) if monthly_limit > 0 else 0

    return {
        "category": category,
        "monthly_limit": monthly_limit,
        "spent": round(spent, 2),
        "remaining": round(remaining, 2),
        "percent_used": round(percent_used, 1),
        "user_override": bool(user_override),
    }


def detect_anomalies(receipt_id):
    """Find items in a receipt priced significantly above historical average."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, item_name, price, category FROM line_items WHERE receipt_id = ?",
        (receipt_id,),
    )
    receipt_items = [dict(r) for r in cursor.fetchall()]

    anomalies = []
    for item in receipt_items:
        cursor.execute(
            """SELECT AVG(price) as avg_price, COUNT(*) as count
               FROM line_items li
               JOIN receipts r ON li.receipt_id = r.id
               WHERE li.category = ? AND li.receipt_id != ?
                 AND r.receipt_date >= date('now', '-90 days')""",
            (item["category"], receipt_id),
        )
        hist = cursor.fetchone()

        if hist and hist["count"] and hist["count"] >= 3:
            avg_price = hist["avg_price"]
            if item["price"] > avg_price * 1.5:
                anomalies.append({
                    "line_item_id": item["id"],
                    "item": item["item_name"],
                    "price": item["price"],
                    "avg_price": round(avg_price, 2),
                    "ratio": round(item["price"] / avg_price, 1),
                    "category": item["category"],
                })

    conn.close()
    return anomalies


def get_trends(category, period="weekly"):
    """Get spending trends over time for a category."""
    conn = get_connection()
    cursor = conn.cursor()

    if period == "weekly":
        group_format = "%Y-W%W"
        lookback = "-12 weeks"
    else:
        group_format = "%Y-%m"
        lookback = "-6 months"

    cursor.execute(
        """SELECT strftime(?, r.receipt_date) as period,
                  SUM(li.price) as total_spent,
                  COUNT(DISTINCT r.id) as receipt_count
           FROM line_items li
           JOIN receipts r ON li.receipt_id = r.id
           WHERE li.category = ?
             AND r.receipt_date >= date('now', ?)
           GROUP BY period
           ORDER BY period""",
        (group_format, category, lookback),
    )

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def set_budget(category, amount):
    """Set or update monthly budget limit for a category (user-initiated)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO budgets (category, monthly_limit, user_override, updated_at)
           VALUES (?, ?, 1, CURRENT_TIMESTAMP)
           ON CONFLICT(category)
           DO UPDATE SET monthly_limit = ?, user_override = 1, updated_at = CURRENT_TIMESTAMP""",
        (category, amount, amount),
    )
    conn.commit()
    conn.close()
    return {"category": category, "monthly_limit": amount, "status": "updated"}


# --- New Agentic Tool Functions ---

def recategorize_item(line_item_id, new_category):
    """Agent recategorizes an item it thinks was miscategorized."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get current category
    cursor.execute(
        "SELECT item_name, category FROM line_items WHERE id = ?", (line_item_id,)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"error": f"Line item {line_item_id} not found"}

    old_category = row["category"]
    item_name = row["item_name"]

    # Update the item
    cursor.execute(
        "UPDATE line_items SET category = ? WHERE id = ?",
        (new_category, line_item_id),
    )

    # Save correction for future learning
    cursor.execute(
        """INSERT INTO category_corrections (item_name, old_category, new_category)
           VALUES (?, ?, ?)""",
        (item_name, old_category, new_category),
    )

    conn.commit()
    conn.close()
    return {
        "item": item_name,
        "old_category": old_category,
        "new_category": new_category,
        "status": "recategorized",
    }


def flag_receipt(receipt_id, reason, line_item_id=None):
    """Agent flags a receipt or item for user review."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO flags (receipt_id, line_item_id, reason)
           VALUES (?, ?, ?)""",
        (receipt_id, line_item_id, reason),
    )
    flag_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"flag_id": flag_id, "status": "flagged", "reason": reason}


def resolve_flag(flag_id):
    """User resolves/dismisses a flag."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE flags SET resolved = 1 WHERE id = ?", (flag_id,))
    conn.commit()
    conn.close()


def get_open_flags(limit=20):
    """Get unresolved flags for display."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT f.id, f.receipt_id, f.line_item_id, f.reason, f.created_at,
                  r.store_name, li.item_name, li.price
           FROM flags f
           LEFT JOIN receipts r ON f.receipt_id = r.id
           LEFT JOIN line_items li ON f.line_item_id = li.id
           WHERE f.resolved = 0
           ORDER BY f.created_at DESC
           LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def create_savings_goal(name, target_amount, deadline=None):
    """Agent creates a savings goal."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO goals (name, target_amount, deadline)
           VALUES (?, ?, ?)""",
        (name, target_amount, deadline),
    )
    goal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"goal_id": goal_id, "name": name, "target": target_amount, "status": "created"}


def get_active_goals():
    """Get all active savings goals."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, name, target_amount, current_saved, deadline, status, created_at
           FROM goals WHERE status = 'active'
           ORDER BY created_at DESC""",
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def log_action(action, detail, receipt_id=None):
    """Log an action the agent took (not just an observation)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO agent_actions (action, detail, receipt_id)
           VALUES (?, ?, ?)""",
        (action, detail, receipt_id),
    )
    conn.commit()
    conn.close()
    return {"status": "logged", "action": action}


def get_recent_actions(limit=20):
    """Get recent agent actions for display."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT action, detail, receipt_id, created_at
           FROM agent_actions
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def compare_merchant_prices(item_name):
    """Find which stores sell a similar item and at what price."""
    conn = get_connection()
    cursor = conn.cursor()
    # Fuzzy match: search for items containing the keyword
    cursor.execute(
        """SELECT r.store_name, li.item_name, AVG(li.price) as avg_price,
                  MIN(li.price) as min_price, MAX(li.price) as max_price,
                  COUNT(*) as times_bought
           FROM line_items li
           JOIN receipts r ON li.receipt_id = r.id
           WHERE LOWER(li.item_name) LIKE ?
           GROUP BY r.store_name
           ORDER BY avg_price ASC""",
        (f"%{item_name.lower()}%",),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows if rows else [{"message": f"No purchase history found for '{item_name}'"}]


def suggest_budgets():
    """Analyze last 60 days of spending and suggest realistic budgets per category."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get avg monthly spending per category over last 60 days (2 months)
    cursor.execute(
        """SELECT li.category,
                  SUM(li.price) as total_60d,
                  COUNT(*) as item_count,
                  COUNT(DISTINCT r.id) as receipt_count
           FROM line_items li
           JOIN receipts r ON li.receipt_id = r.id
           WHERE r.receipt_date >= date('now', '-60 days')
           GROUP BY li.category"""
    )
    spending = [dict(r) for r in cursor.fetchall()]

    # Get current budgets
    cursor.execute("SELECT category, monthly_limit, user_override FROM budgets")
    budgets = {row["category"]: dict(row) for row in cursor.fetchall()}
    conn.close()

    suggestions = []
    for row in spending:
        cat = row["category"]
        monthly_avg = row["total_60d"] / 2  # 60 days ≈ 2 months
        suggested = round(monthly_avg * 1.15, -1)  # +15% buffer, round to nearest 10

        current = budgets.get(cat, {})
        current_limit = current.get("monthly_limit", 0)
        user_set = current.get("user_override", 0)

        suggestions.append({
            "category": cat,
            "monthly_avg_spending": round(monthly_avg, 2),
            "suggested_limit": suggested,
            "current_limit": current_limit,
            "user_override": bool(user_set),
            "item_count": row["item_count"],
        })

    return suggestions


def auto_adjust_budget(category, new_limit, reason):
    """Agent adjusts a budget — only if user hasn't manually overridden it."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if user has overridden this budget
    cursor.execute(
        "SELECT user_override FROM budgets WHERE category = ?", (category,)
    )
    row = cursor.fetchone()
    if row and row["user_override"]:
        conn.close()
        return {
            "category": category,
            "status": "skipped",
            "reason": "User has manually set this budget — not overriding",
        }

    cursor.execute(
        """INSERT INTO budgets (category, monthly_limit, user_override, updated_at)
           VALUES (?, ?, 0, CURRENT_TIMESTAMP)
           ON CONFLICT(category)
           DO UPDATE SET monthly_limit = ?, updated_at = CURRENT_TIMESTAMP
           WHERE user_override = 0""",
        (category, new_limit, new_limit),
    )

    # Log the action
    cursor.execute(
        """INSERT INTO agent_actions (action, detail)
           VALUES (?, ?)""",
        ("auto_adjust_budget", f"{category}: set to ${new_limit:.0f} — {reason}"),
    )

    conn.commit()
    conn.close()
    return {
        "category": category,
        "new_limit": new_limit,
        "reason": reason,
        "status": "adjusted",
    }


def get_category_corrections(item_name=None):
    """Get category corrections, optionally filtered by item name."""
    conn = get_connection()
    cursor = conn.cursor()
    if item_name:
        cursor.execute(
            """SELECT new_category, COUNT(*) as count
               FROM category_corrections
               WHERE LOWER(item_name) = LOWER(?)
               GROUP BY new_category
               ORDER BY count DESC
               LIMIT 1""",
            (item_name,),
        )
    else:
        cursor.execute(
            """SELECT item_name, old_category, new_category, created_at
               FROM category_corrections
               ORDER BY created_at DESC
               LIMIT 20"""
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# --- Dashboard Queries ---

def get_all_budgets():
    """Get all budget categories with current spending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT b.category, b.monthly_limit, b.user_override,
                  COALESCE(monthly.spent, 0) as spent
           FROM budgets b
           LEFT JOIN (
               SELECT li.category, SUM(li.price) as spent
               FROM line_items li
               JOIN receipts r ON li.receipt_id = r.id
               WHERE strftime('%Y-%m', r.receipt_date) = strftime('%Y-%m', 'now', 'localtime')
               GROUP BY li.category
           ) monthly ON b.category = monthly.category
           ORDER BY b.category""",
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_recent_receipts(limit=20):
    """Get recent receipts with their items."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT r.id, r.store_name, r.receipt_date, r.total,
                  GROUP_CONCAT(li.item_name || ' ($' || li.price || ')', ', ') as items
           FROM receipts r
           LEFT JOIN line_items li ON li.receipt_id = r.id
           GROUP BY r.id
           ORDER BY r.receipt_date DESC
           LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_spending_by_category_this_month():
    """Get spending breakdown by category for current month."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT li.category, SUM(li.price) as total
           FROM line_items li
           JOIN receipts r ON li.receipt_id = r.id
           WHERE strftime('%Y-%m', r.receipt_date) = strftime('%Y-%m', 'now', 'localtime')
           GROUP BY li.category
           ORDER BY total DESC""",
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_daily_spending(days=30):
    """Get daily spending totals."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT r.receipt_date as day, SUM(li.price) as total
           FROM line_items li
           JOIN receipts r ON li.receipt_id = r.id
           WHERE r.receipt_date >= date('now', ?)
           GROUP BY r.receipt_date
           ORDER BY r.receipt_date""",
        (f"-{days} days",),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_agent_memories(limit=10):
    """Get recent agent observations for context."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT observation, action_taken, created_at
           FROM agent_memory
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# Initialize DB on import
init_db()
