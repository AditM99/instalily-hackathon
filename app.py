
"""Receipt Spending Agent — Streamlit UI"""

import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import db
from ocr import process_receipt_image, crop_region
from categorizer import categorize_items
import agent

st.set_page_config(
    page_title="Receipt Spending Agent",
    page_icon="🧾",
    layout="wide",
)

# Initialize DB
db.init_db()

# Session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


CAT_EMOJI = {
    "groceries": "🛒", "dining": "🍔", "transport": "🚗",
    "entertainment": "🎬", "health": "💊", "clothing": "👕",
    "utilities": "🔌", "other": "📦",
}


def render_sidebar():
    """Render budget status in sidebar."""
    st.sidebar.title("Budget Status")
    budgets = db.get_all_budgets()

    for b in budgets:
        limit = b["monthly_limit"]
        spent = b["spent"]
        pct = min(spent / limit * 100, 100) if limit > 0 else 0

        badge = "auto" if not b.get("user_override") else "manual"
        st.sidebar.markdown(f"**{b['category'].title()}** `{badge}`")
        st.sidebar.progress(pct / 100, text=f"${spent:.0f} / ${limit:.0f}")

    st.sidebar.divider()
    if st.sidebar.button("Seed Demo Data", help="Populate 30 days of fake data"):
        from seed_demo_data import seed
        seed()
        st.rerun()

    if st.sidebar.button("Reset Database", help="Clear all data"):
        import os
        if os.path.exists(db.DB_PATH):
            os.remove(db.DB_PATH)
        db.init_db()
        st.rerun()


def tab_scan():
    """Tab 1: Scan Receipt."""
    st.header("Scan a Receipt")

    col1, col2 = st.columns([1, 1])

    with col1:
        upload_method = st.radio(
            "Input method", ["Upload Image", "Camera"], horizontal=True
        )

        img = None
        if upload_method == "Camera":
            img_file = st.camera_input("Take a photo of your receipt")
            if img_file:
                img = Image.open(img_file)
        else:
            img_file = st.file_uploader(
                "Upload receipt image", type=["png", "jpg", "jpeg", "webp"]
            )
            if img_file:
                img = Image.open(img_file)

        if img:
            st.image(img, caption="Receipt", width="stretch")

    with col2:
        if img:
            with st.spinner("Reading receipt with OCR..."):
                parsed, preprocessed = process_receipt_image(img)

            # Show raw OCR for debugging
            with st.expander("Raw OCR Output (debug)", expanded=False):
                st.text(parsed["raw_text"])

            # Show parsed data
            st.subheader("Parsed Receipt")
            st.write(f"**Store:** {parsed['store_name']}")
            st.write(f"**Date:** {parsed['date'] or 'Not detected'}")

            if parsed["items"]:
                # Categorize items
                with st.spinner("Categorizing items..."):
                    categorized = categorize_items(parsed["items"])

                categories = list(CAT_EMOJI.keys())

                # Low-confidence OCR warnings
                if parsed["low_confidence"]:
                    st.warning(
                        f"Agent is unsure about {len(parsed['low_confidence'])} "
                        f"text region(s). Review items below and correct if needed."
                    )

                # Editable items table — user can fix names and categories
                st.subheader("Items — review & edit")
                for i, item in enumerate(categorized):
                    c1, c2, c3 = st.columns([3, 1.5, 2])
                    with c1:
                        new_name = st.text_input(
                            "Item", value=item["name"],
                            key=f"item_name_{i}", label_visibility="collapsed",
                        )
                        categorized[i]["name"] = new_name
                    with c2:
                        st.markdown(f"**${item['price']:.2f}**")
                    with c3:
                        current_cat = item.get("category", "other")
                        cat_idx = categories.index(current_cat) if current_cat in categories else len(categories) - 1
                        new_cat = st.selectbox(
                            "Category", categories, index=cat_idx,
                            key=f"item_cat_{i}", label_visibility="collapsed",
                            format_func=lambda c: f"{CAT_EMOJI.get(c, '📦')} {c}",
                        )
                        categorized[i]["category"] = new_cat

                if parsed["total"]:
                    st.write(f"**Total:** ${parsed['total']:.2f}")
                if parsed["tax"]:
                    st.write(f"**Tax:** ${parsed['tax']:.2f}")

                # Save to database
                st.divider()
                if st.button("Save Receipt & Analyze", type="primary"):
                    from datetime import date as _date
                    with st.spinner("Saving..."):
                        receipt_id = db.insert_receipt(
                            store_name=parsed["store_name"],
                            receipt_date=parsed["date"] or str(_date.today()),
                            total=parsed["total"] or sum(
                                i["price"] for i in categorized
                            ),
                            tax=parsed["tax"] or 0,
                            image_path=None,
                            raw_ocr_text=parsed["raw_text"],
                        )

                        for item in categorized:
                            db.insert_line_item(
                                receipt_id=receipt_id,
                                item_name=item["name"],
                                price=item["price"],
                                category=item.get("category", "other"),
                                confidence=item.get("model_confidence", 0.5),
                            )

                            # Save user corrections for future learning
                            if item.get("category") != item.get("original_category"):
                                db.recategorize_item(
                                    line_item_id=0,  # will be ignored, just logs the correction
                                    new_category=item["category"],
                                )

                    # Run rule engine — instant, no LLM needed
                    st.divider()
                    rule_actions = agent.run_rules(receipt_id)

                    if rule_actions:
                        st.subheader("Agent Actions")
                        for ra in rule_actions:
                            if ra["type"] == "flag":
                                st.warning(ra["detail"])
                            elif ra["type"] == "budget_adjust":
                                st.info(ra["detail"])
                            elif ra["type"] == "warning":
                                st.error(ra["detail"])
                            elif ra["type"] == "goal":
                                st.success(ra["detail"])
                    else:
                        st.success("Receipt saved. No issues detected.")

            else:
                st.warning("No items detected. Try a clearer image.")


def tab_dashboard():
    """Tab 2: Spending Dashboard."""
    st.header("Spending Dashboard")

    # Flagged items at the top
    flags = db.get_open_flags()
    if flags:
        st.subheader("Flagged by Agent")
        for f in flags:
            col_flag, col_btn = st.columns([4, 1])
            with col_flag:
                item_info = f""
                if f.get("item_name"):
                    item_info = f" — {f['item_name']} (${f.get('price', 0):.2f})"
                st.warning(
                    f"**{f.get('store_name', 'Unknown')}**{item_info}: {f['reason']}"
                )
            with col_btn:
                if st.button("Dismiss", key=f"dismiss_flag_{f['id']}"):
                    db.resolve_flag(f["id"])
                    st.rerun()
        st.divider()

    col1, col2 = st.columns(2)

    with col1:
        spending = db.get_spending_by_category_this_month()
        if spending:
            df = pd.DataFrame(spending)
            fig = px.pie(
                df,
                values="total",
                names="category",
                title="Spending by Category (This Month)",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No spending data yet. Scan a receipt or seed demo data.")

    with col2:
        daily = db.get_daily_spending(30)
        if daily:
            df = pd.DataFrame(daily)
            fig = px.bar(
                df,
                x="day",
                y="total",
                title="Daily Spending (Last 30 Days)",
                color_discrete_sequence=["#4CAF50"],
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Amount ($)")
            st.plotly_chart(fig, width="stretch")

    # Trend lines by category
    st.subheader("Category Trends")
    categories = [
        "groceries", "dining", "transport", "entertainment",
        "health", "clothing", "utilities", "other",
    ]
    selected_cat = st.selectbox("Select category", categories)

    trends = db.get_trends(selected_cat, "weekly")
    if trends:
        df = pd.DataFrame(trends)
        fig = px.line(
            df,
            x="period",
            y="total_spent",
            title=f"{selected_cat.title()} Spending Trend (Weekly)",
            markers=True,
        )
        fig.update_layout(xaxis_title="Week", yaxis_title="Amount ($)")
        st.plotly_chart(fig, width="stretch")

    # Recent agent actions
    st.subheader("Recent Agent Actions")
    actions = db.get_recent_actions(10)
    if actions:
        for a in actions:
            st.caption(f"**{a['action']}** — {a['detail']} ({a['created_at']})")
    else:
        st.write("No agent actions yet.")

    # Recent receipts
    st.subheader("Recent Receipts")
    receipts = db.get_recent_receipts(10)
    if receipts:
        df = pd.DataFrame(receipts)
        df = df[["store_name", "receipt_date", "total", "items"]]
        df.columns = ["Store", "Date", "Total ($)", "Items"]
        st.dataframe(df, width="stretch")

    # Export spending data
    if spending:
        st.divider()
        export_df = pd.DataFrame(spending)
        if daily:
            daily_df = pd.DataFrame(daily)
            csv = export_df.to_csv(index=False) + "\n\nDaily Spending\n" + daily_df.to_csv(index=False)
        else:
            csv = export_df.to_csv(index=False)
        st.download_button(
            "Download Spending Report (CSV)",
            csv, "spending_report.csv", "text/csv",
        )


def _run_chat(prompt):
    """Send a prompt to the agent and store the response in session state."""
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    # Build history for agent context (last 4 messages)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_messages[:-1]
    ]

    response, tool_calls = agent.chat(prompt, history)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": response, "tool_calls": tool_calls}
    )


def tab_chat():
    """Tab 3: Ask Agent."""
    st.header("Ask Your Spending Agent")
    st.caption(
        "Ask questions about your spending — or tell the agent to take actions "
        "(adjust budgets, set goals, flag items)."
    )

    # Handle pending suggestion click
    if st.session_state.get("pending_chat_prompt"):
        prompt = st.session_state.pop("pending_chat_prompt")
        _run_chat(prompt)

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                with st.expander(f"Agent used {len(msg['tool_calls'])} tools"):
                    for tc in msg["tool_calls"]:
                        st.code(f"{tc['tool']}({tc['args']})")

    # Chat input
    if prompt := st.chat_input("Ask about your spending..."):
        _run_chat(prompt)
        st.rerun()

    # Suggested questions
    if not st.session_state.chat_messages:
        st.markdown("**Try asking:**")
        suggestions = [
            "How much did I spend on dining this month?",
            "Am I on track with my grocery budget?",
            "Set my dining budget to $200",
            "Create a savings goal of $500 for vacation",
        ]
        for s in suggestions:
            if st.button(s, key=f"suggest_{s}"):
                st.session_state.pending_chat_prompt = s
                st.rerun()


def tab_budgets():
    """Tab 4: Budget Manager."""
    st.header("Budget Manager")

    budgets = db.get_all_budgets()

    if not budgets:
        st.info("No budgets set. Default budgets will be created.")
        return

    cols = st.columns(2)
    for i, b in enumerate(budgets):
        with cols[i % 2]:
            limit = b["monthly_limit"]
            spent = b["spent"]
            remaining = limit - spent
            pct = min(spent / limit * 100, 100) if limit > 0 else 0

            status_color = "🟢" if pct < 70 else "🟡" if pct < 90 else "🔴"
            managed_by = "you" if b.get("user_override") else "agent (auto)"

            st.markdown(f"### {status_color} {b['category'].title()}")
            st.caption(f"Managed by: **{managed_by}**")
            st.progress(pct / 100)
            st.write(f"Spent: **${spent:.2f}** / ${limit:.2f}")
            st.write(f"Remaining: **${remaining:.2f}**")

            new_limit = st.number_input(
                f"Update {b['category']} budget",
                value=float(limit),
                min_value=0.0,
                step=50.0,
                key=f"budget_{b['category']}",
            )
            if new_limit != limit:
                if st.button(f"Save", key=f"save_{b['category']}"):
                    db.set_budget(b["category"], new_limit)  # marks user_override=1
                    st.rerun()

            st.divider()

    # Savings goals
    st.subheader("Savings Goals")

    # Create new goal form
    with st.expander("Create a new savings goal", expanded=False):
        gc1, gc2 = st.columns([2, 1])
        with gc1:
            goal_name = st.text_input("Goal name", placeholder="e.g. Vacation, Emergency Fund")
        with gc2:
            goal_amount = st.number_input("Target ($)", min_value=1.0, value=500.0, step=50.0)
        if st.button("Create Goal", type="primary"):
            if goal_name.strip():
                db.create_savings_goal(goal_name.strip(), goal_amount)
                st.success(f"Created goal **{goal_name}** — ${goal_amount:.0f}")
                st.rerun()
            else:
                st.error("Please enter a goal name.")

    goals = db.get_active_goals()
    if goals:
        for g in goals:
            pct = min(g["current_saved"] / g["target_amount"] * 100, 100) if g["target_amount"] > 0 else 0
            st.markdown(f"**{g['name']}** — ${g['current_saved']:.0f} / ${g['target_amount']:.0f}")
            st.progress(pct / 100)
            if g.get("deadline"):
                st.caption(f"Deadline: {g['deadline']}")
    else:
        st.write("No savings goals yet. Create one above or ask the agent!")

    # Agent actions log
    st.subheader("Agent Action Log")
    actions = db.get_recent_actions(10)
    if actions:
        for a in actions:
            st.info(f"**{a['action']}**: {a['detail']} ({a['created_at']})")
    else:
        st.write("No agent actions yet.")

    # Agent memories
    st.subheader("Agent Observations")
    memories = db.get_agent_memories(limit=10)
    if memories:
        for m in memories:
            st.caption(f"**{m['created_at']}**: {m['observation']}")


# --- Main Layout ---
st.title("Receipt Spending Agent")
st.caption("On-device AI that watches your spending and manages your money.")

render_sidebar()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📷 Scan Receipt", "📊 Dashboard", "💬 Ask Agent", "💰 Budgets"]
)

with tab1:
    tab_scan()

with tab2:
    tab_dashboard()

with tab3:
    tab_chat()

with tab4:
    tab_budgets()
