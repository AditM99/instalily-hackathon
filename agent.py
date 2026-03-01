"""Agentic system for receipt spending analysis.

Rule engine: runs deterministic decisions instantly (no LLM).
Chat: Ollama LLM agent with tools — calls them only when needed.
"""

import json
import re
import ollama
import db

MODEL_NAME = "mistral:latest"
CATEGORIES = ["groceries", "dining", "transport", "entertainment",
              "health", "clothing", "utilities", "other"]

# ---------------------------------------------------------------------------
# RULE ENGINE — runs on every receipt, pure Python, instant
# ---------------------------------------------------------------------------

def run_rules(receipt_id):
    """
    Autonomous rule engine. Runs all decision rules against a receipt.
    Returns list of actions taken (dicts with action, detail, etc).
    """
    actions = []

    # --- Rule 1: Anomaly detection → flag ---
    anomalies = db.detect_anomalies(receipt_id)
    for a in anomalies:
        if a["ratio"] >= 2.0:
            db.flag_receipt(receipt_id,
                f"Price anomaly: {a['item']} costs ${a['price']:.2f} but avg is ${a['avg_price']:.2f} ({a['ratio']}x)",
                line_item_id=a.get("line_item_id"))
            db.log_action("flagged_anomaly",
                f"{a['item']}: ${a['price']:.2f} is {a['ratio']}x the avg ${a['avg_price']:.2f}",
                receipt_id)
            actions.append({
                "type": "flag",
                "detail": f"Flagged **{a['item']}** — ${a['price']:.2f} is {a['ratio']}x the category average (${a['avg_price']:.2f})",
            })

    # --- Rule 2: Auto-adjust budgets from spending patterns ---
    suggestions = db.suggest_budgets()
    for s in suggestions:
        cat = s["category"]
        if (s["item_count"] >= 5
                and not s["user_override"]
                and s["suggested_limit"] > 0
                and abs(s["suggested_limit"] - s["current_limit"]) / max(s["current_limit"], 1) > 0.2):
            result = db.auto_adjust_budget(
                cat, s["suggested_limit"],
                f"Based on avg monthly spending of ${s['monthly_avg_spending']:.0f} (+15% buffer)")
            if result.get("status") == "adjusted":
                actions.append({
                    "type": "budget_adjust",
                    "detail": f"Auto-set **{cat}** budget to ${s['suggested_limit']:.0f} (avg spend: ${s['monthly_avg_spending']:.0f}/mo)",
                })

    # --- Rule 3: Budget warnings (>90% used) ---
    all_budgets = db.get_all_budgets()
    for b in all_budgets:
        if b["monthly_limit"] > 0:
            pct = b["spent"] / b["monthly_limit"] * 100
            if pct >= 90:
                db.log_action("budget_warning",
                    f"{b['category']}: {pct:.0f}% used (${b['spent']:.0f}/${b['monthly_limit']:.0f})",
                    receipt_id)
                actions.append({
                    "type": "warning",
                    "detail": f"**{b['category'].title()}** is at {pct:.0f}% of budget (${b['spent']:.0f} / ${b['monthly_limit']:.0f})",
                })

    # --- Rule 4: Savings goal suggestion (<50% budget used with enough history) ---
    for b in all_budgets:
        if b["monthly_limit"] > 0 and b["spent"] > 0:
            pct = b["spent"] / b["monthly_limit"] * 100
            if pct < 50:
                existing_goals = db.get_active_goals()
                has_goal = any(b["category"] in g["name"].lower() for g in existing_goals)
                if not has_goal:
                    saved = b["monthly_limit"] - b["spent"]
                    db.create_savings_goal(
                        f"{b['category'].title()} savings",
                        round(saved * 3, -1),
                    )
                    db.log_action("created_goal",
                        f"Savings goal for {b['category']}: ${saved:.0f}/mo potential",
                        receipt_id)
                    actions.append({
                        "type": "goal",
                        "detail": f"Created savings goal for **{b['category'].title()}** — you're only using {pct:.0f}% of budget, could save ~${saved:.0f}/mo",
                    })

    if actions:
        summary = f"Took {len(actions)} actions on receipt #{receipt_id}"
        db.insert_agent_memory(summary, "rule_engine", receipt_id)

    return actions


# ---------------------------------------------------------------------------
# CHAT — agentic LLM with tools, calls them only when needed
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """You are a spending assistant. When you need data, reply with ONLY a JSON tool call.

Tools:
1. query_spending(category, days) - get spending. category: groceries/dining/transport/entertainment/health/clothing/utilities/other. days: 1=today, 7=this week, 30=this month (default).
2. check_budget(category) - check budget status.
3. set_budget(category, amount) - change a budget limit.
4. create_savings_goal(name, target_amount) - create a savings goal.

Examples:
User: "How much did I spend on dining today?" -> {"tool":"query_spending","args":{"category":"dining","days":1}}
User: "What did I spend this week?" -> {"tool":"query_spending","args":{"days":7}}
User: "Am I on track with groceries?" -> {"tool":"check_budget","args":{"category":"groceries"}}
User: "Set dining budget to 250" -> {"tool":"set_budget","args":{"category":"dining","amount":250}}
User: "Create a savings goal of 500 for vacation" -> {"tool":"create_savings_goal","args":{"name":"Vacation","target_amount":500}}

After receiving tool results, give a short helpful answer."""

_CHAT_TOOL_MAP = {
    "query_spending": db.query_spending,
    "check_budget": db.check_budget,
    "set_budget": db.set_budget,
    "create_savings_goal": db.create_savings_goal,
}


def _parse_tool_call(content):
    """Extract tool call JSON from agent response."""
    content = content.strip()

    # Look for ```tool block first
    match = re.search(r"```(?:tool)?\s*\n?(.+?)\n?```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find JSON object containing "tool" key anywhere in content
    # Walk through and find balanced braces
    for m in re.finditer(r'\{', content):
        start = m.start()
        depth = 0
        for j in range(start, len(content)):
            if content[j] == '{':
                depth += 1
            elif content[j] == '}':
                depth -= 1
            if depth == 0:
                candidate = content[start:j+1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                break

    return None


def _format_tool_result(tool_name, args, result):
    """Format tool result into a human-readable answer. No LLM needed."""
    if tool_name == "query_spending":
        if not result:
            cat = args.get("category")
            days = args.get("days", 30)
            period = {1: "today", 7: "this week", 30: "this month"}.get(days, f"in the last {days} days")
            return f"No spending found{f' on **{cat}**' if cat else ''} {period}."
        lines = []
        total = 0
        for r in result:
            lines.append(f"- **{r['category'].title()}**: ${r['total_spent']:.2f} ({r['item_count']} items)")
            total += r["total_spent"]
        days = args.get("days", 30)
        period = {1: "today", 7: "this week", 30: "this month"}.get(days, f"in the last {days} days")
        header = f"Spending {period}:"
        if len(result) > 1:
            lines.append(f"\n**Total: ${total:.2f}**")
        return header + "\n" + "\n".join(lines)

    elif tool_name == "check_budget":
        if isinstance(result, dict) and "error" not in result:
            cat = result["category"].title()
            pct = result["percent_used"]
            status = "on track" if pct < 80 else "over budget" if pct > 100 else "close to your limit"
            return (
                f"**{cat} Budget**: ${result['spent']:.2f} / ${result['monthly_limit']:.2f} "
                f"({pct:.0f}% used)\n\n"
                f"Remaining: **${result['remaining']:.2f}** — you're {status}."
            )
        return f"No budget found for **{args.get('category', 'unknown')}**."

    elif tool_name == "set_budget":
        if isinstance(result, dict) and result.get("status") == "updated":
            return f"Done! **{result['category'].title()}** budget set to **${result['monthly_limit']:.2f}**/month."
        return f"Budget updated."

    elif tool_name == "create_savings_goal":
        if isinstance(result, dict) and result.get("status") == "created":
            return f"Created savings goal **{result['name']}** with target **${result['target']:.2f}**. Track it in the Budgets tab!"
        return f"Savings goal created."

    return json.dumps(result, indent=2, default=str)


def chat(user_message, chat_history=None, max_steps=2):
    """
    Agentic chat — LLM decides whether to use tools or answer directly.
    Max 1 tool call + 1 answer = 2 LLM calls worst case.
    """
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history[-4:])

    messages.append({"role": "user", "content": user_message})

    tool_calls_log = []

    for step in range(max_steps):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": 0, "num_predict": 100},
            )
            content = response["message"]["content"]
        except Exception as e:
            return f"Agent error: {str(e)}", tool_calls_log

        tool_call = _parse_tool_call(content)

        if tool_call is None:
            # No tool call — this is the final answer
            return content, tool_calls_log

        tool_name = tool_call.get("tool", "")
        if tool_name not in _CHAT_TOOL_MAP:
            return content, tool_calls_log

        args = tool_call.get("args", {})

        # Match category from user message (handles partial matches like "grocery" -> "groceries")
        lower_msg = user_message.lower()
        _CAT_ALIASES = {
            "grocery": "groceries", "groceries": "groceries",
            "dining": "dining", "food": "dining", "restaurant": "dining",
            "transport": "transport", "travel": "transport", "gas": "transport",
            "entertainment": "entertainment", "fun": "entertainment",
            "health": "health", "medical": "health", "pharmacy": "health",
            "clothing": "clothing", "clothes": "clothing",
            "utilities": "utilities", "utility": "utilities",
            "other": "other",
        }
        detected_cat = None
        for alias, canonical in _CAT_ALIASES.items():
            if alias in lower_msg:
                detected_cat = canonical
                break

        # If query_spending but user asked about budget, redirect to check_budget
        if tool_name == "query_spending" and any(w in lower_msg for w in ["budget", "on track", "remaining", "left"]):
            if detected_cat:
                tool_name = "check_budget"
                args = {"category": detected_cat}

        # If query_spending without category but user mentioned one, inject it
        if tool_name == "query_spending" and "category" not in args and detected_cat:
            args["category"] = detected_cat

        try:
            result = _CHAT_TOOL_MAP[tool_name](**args)
        except Exception as e:
            result = {"error": str(e)}

        tool_calls_log.append({
            "tool": tool_name,
            "args": args,
            "result": result,
        })

        # Format answer directly — no second LLM call needed
        # This avoids hallucinated numbers from the small model
        answer = _format_tool_result(tool_name, args, result)
        return answer, tool_calls_log

    return content, tool_calls_log


def get_clarification_questions(low_confidence_items):
    """Generate clarification questions for low-confidence OCR results."""
    questions = []
    for item in low_confidence_items:
        questions.append({
            "original_text": item["text"],
            "confidence": item["confidence"],
            "bbox": item.get("bbox"),
            "question": f"I'm not sure about this item: \"{item['text']}\" (confidence: {item['confidence']:.0%}). Is this correct?",
        })
    return questions
