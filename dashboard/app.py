from __future__ import annotations
# dashboard/app.py
#
# Streamlit Reviewer Dashboard — the human-in-the-loop workbench.
#
# Screens:
#   1. Alert Queue    — list of open HIGH/MEDIUM cases
#   2. Case Review    — evidence panel for a single case
#   3. Audit Log      — immutable event chain viewer
#
# Run locally:
#   streamlit run dashboard/app.py
#
# Deploy to Hugging Face Spaces:
#   - Connect repo → Space auto-deploys on push
#   - Set APP_ENV=production in Space secrets
import sys
import os

# Load .env into os.environ BEFORE importing any prediction modules —
# explainer.py and judge.py call os.getenv("GEMINI_API_KEY") / os.getenv("GROQ_API_KEY")
# directly, and pydantic Settings(env_file=".env") does NOT populate os.environ.
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
from datetime import datetime, timezone, timedelta

import streamlit as st

# ── Page config — MUST be the first Streamlit command, before any other
# imports that might emit output or touch the Streamlit context ──────────
st.set_page_config(
    page_title="Compliance Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.schemas.transaction_event import TransactionEvent, RiskClass
from features.redis_client import FeatureStoreClient
from prediction.orchestrator import process_transaction, UnifiedRiskOutput

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high   { color: #dc2626; font-weight: 700; font-size: 1.1rem; }
    .risk-medium { color: #d97706; font-weight: 700; font-size: 1.1rem; }
    .risk-low    { color: #16a34a; font-weight: 700; font-size: 1.1rem; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .audit-row {
        font-family: monospace; font-size: 0.75rem;
        padding: 4px 0; border-bottom: 1px solid #f1f5f9;
    }
    .badge-escalate { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .badge-flag     { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .badge-monitor  { background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .badge-close    { background:#dcfce7; color:#166534; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────
if "cases" not in st.session_state:
    st.session_state.cases = []          # list of UnifiedRiskOutput dicts
if "audit_events" not in st.session_state:
    st.session_state.audit_events = []   # list of audit event dicts
if "store" not in st.session_state:
    st.session_state.store = FeatureStoreClient()
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None


# ── Helpers ───────────────────────────────────────────────────────────────

def risk_badge(risk_class: str) -> str:
    colours = {
        "HIGH":   ("🔴", "#dc2626"),
        "MEDIUM": ("🟡", "#d97706"),
        "LOW":    ("🟢", "#16a34a"),
    }
    icon, _ = colours.get(risk_class, ("⚪", "#6b7280"))
    return f"{icon} {risk_class}"


def recommendation_badge(rec: str) -> str:
    badges = {
        "ESCALATE": "badge-escalate",
        "FLAG":     "badge-flag",
        "MONITOR":  "badge-monitor",
        "CLOSE":    "badge-close",
    }
    css = badges.get(rec, "badge-monitor")
    return f'<span class="{css}">{rec}</span>'


def shap_waterfall(top_features: list[dict], title: str = "SHAP Feature Contributions") -> go.Figure:
    """Plotly waterfall chart of SHAP values."""
    features     = [f["feature"] for f in top_features]
    contributions = [f["contribution"] for f in top_features]
    values        = [f["value"] for f in top_features]

    colours = ["#dc2626" if c > 0 else "#16a34a" for c in contributions]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=[f"{feat}<br><i>val={val}</i>" for feat, val in zip(features, values)],
        orientation="h",
        marker_color=colours,
        text=[f"{c:+.3f}" for c in contributions],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="SHAP contribution (+ raises risk, − lowers risk)",
        height=max(300, len(features) * 55),
        margin=dict(l=10, r=60, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=1.5)
    return fig


def score_gauge(risk_score: float, risk_class: str) -> go.Figure:
    """Plotly gauge chart for risk score."""
    colour = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}.get(risk_class, "#6b7280")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score * 100, 1),
        title={"text": "Risk Score", "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": colour, "thickness": 0.3},
            "steps": [
                {"range": [0, 45],  "color": "#dcfce7"},
                {"range": [45, 70], "color": "#fef3c7"},
                {"range": [70, 100],"color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 4},
                "thickness": 0.75,
                "value": risk_score * 100,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def run_pipeline(event: TransactionEvent) -> UnifiedRiskOutput:
    """Runs the async orchestrator synchronously for Streamlit."""
    store = st.session_state.store
    store.seed_customer(
        event.customer_id,
        avg_amount=float(st.session_state.get("avg_amount", 5000)),
        risk_rating=int(st.session_state.get("risk_rating", 2)),
        prior_alerts=int(st.session_state.get("prior_alerts", 0)),
        account_age_days=int(st.session_state.get("account_age", 365)),
    )
    return asyncio.run(process_transaction(event, store))


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR — navigation + transaction input
# ════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🛡️ Compliance Risk")
    st.caption("Deloitte POC · The Talent Grid")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Alert Queue", "🔍 Case Review", "📋 Audit Log"],
        label_visibility="collapsed",
    )
    st.divider()

    st.subheader("Score a Transaction")

    with st.form("txn_form"):
        txn_id    = st.text_input("Transaction ID", value="TXN-001")
        cust_id   = st.text_input("Customer ID",    value="CUST-42")
        amount    = st.number_input("Amount (USD)", min_value=1.0, value=95000.0, step=1000.0)
        currency  = st.selectbox("Currency", ["USD", "EUR", "GBP", "AED", "JPY"])
        channel   = st.selectbox("Channel",  ["WIRE", "SWIFT", "ACH", "CARD"])
        dest_jur  = st.selectbox("Destination", ["AE", "RU", "IR", "KP", "BY", "US", "GB", "DE", "JP", "FR"])
        product   = st.selectbox("Product", ["WIRE_TRANSFER", "SWIFT_TRANSFER", "ACH_PAYMENT", "CARD_PAYMENT"])

        st.caption("Customer profile")
        col1, col2 = st.columns(2)
        with col1:
            avg_amount = st.number_input("Avg amount", value=5000.0, step=500.0)
            risk_rating = st.selectbox("Risk rating", [1, 2, 3, 4, 5], index=1)
        with col2:
            prior_alerts = st.number_input("Prior alerts", value=0, min_value=0, step=1)
            account_age  = st.number_input("Acct age (days)", value=365, step=30)

        submitted = st.form_submit_button("▶ Score Transaction", use_container_width=True)

    if submitted:
        st.session_state.avg_amount   = avg_amount
        st.session_state.risk_rating  = risk_rating
        st.session_state.prior_alerts = prior_alerts
        st.session_state.account_age  = account_age

        event = TransactionEvent(
            transaction_id=txn_id,
            customer_id=cust_id,
            amount=amount,
            currency=currency,
            channel=channel,
            origin_account=f"ACC-US-{cust_id}",
            destination_account=f"ACC-{dest_jur}-999",
            jurisdiction_origin="US",
            jurisdiction_destination=dest_jur,
            product_type=product,
            timestamp=datetime.now(timezone.utc),
        )

        with st.spinner("Running pipeline…"):
            try:
                result = run_pipeline(event)
                result_dict = result.model_dump(mode="json")
                result_dict["_scored_at"] = datetime.now(timezone.utc).isoformat()

                # Add to cases list (replace if same txn_id)
                st.session_state.cases = [
                    c for c in st.session_state.cases
                    if c["transaction_id"] != txn_id
                ]
                st.session_state.cases.insert(0, result_dict)
                st.session_state.selected_case = txn_id

                rc = result.risk_class.value
                if rc == "HIGH":
                    st.error(f"🔴 HIGH RISK — Score: {result.risk_score:.3f}")
                elif rc == "MEDIUM":
                    st.warning(f"🟡 MEDIUM RISK — Score: {result.risk_score:.3f}")
                else:
                    st.success(f"🟢 LOW RISK — Score: {result.risk_score:.3f}")

            except Exception as e:
                st.error(f"Pipeline error: {e}")


# ════════════════════════════════════════════════════════════════════════
# PAGE 1: ALERT QUEUE
# ════════════════════════════════════════════════════════════════════════

if page == "🏠 Alert Queue":
    st.title("Alert Queue")

    cases = st.session_state.cases

    # Summary metrics
    total  = len(cases)
    high   = sum(1 for c in cases if c.get("risk_class") == "HIGH")
    medium = sum(1 for c in cases if c.get("risk_class") == "MEDIUM")
    low    = sum(1 for c in cases if c.get("risk_class") == "LOW")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases",   total)
    col2.metric("🔴 High Risk",  high)
    col3.metric("🟡 Medium Risk", medium)
    col4.metric("🟢 Low Risk",   low)
    st.divider()

    if not cases:
        st.info("No cases yet. Score a transaction using the sidebar form.")
    else:
        st.subheader(f"{total} case{'s' if total != 1 else ''}")

        for case in cases:
            rc    = case.get("risk_class", "LOW")
            score = case.get("risk_score", 0)
            txn   = case.get("transaction_id", "?")
            cid   = case.get("case_id") or "—"
            judge = case.get("llm_judge") or {}
            rec   = judge.get("recommendation", "—")

            with st.container():
                c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 1])
                c1.markdown(f"**{txn}**")
                c2.markdown(risk_badge(rc))
                c3.markdown(f"`{score:.3f}`")
                c4.markdown(recommendation_badge(rec), unsafe_allow_html=True)
                if c5.button("Review →", key=f"btn_{txn}"):
                    st.session_state.selected_case = txn
                    st.rerun()
                st.markdown("---")


# ════════════════════════════════════════════════════════════════════════
# PAGE 2: CASE REVIEW
# ════════════════════════════════════════════════════════════════════════

elif page == "🔍 Case Review":
    st.title("Case Review")

    cases = st.session_state.cases
    if not cases:
        st.info("No cases yet. Score a transaction using the sidebar form.")
        st.stop()

    txn_ids = [c["transaction_id"] for c in cases]
    default = txn_ids.index(st.session_state.selected_case) \
        if st.session_state.selected_case in txn_ids else 0

    selected_id = st.selectbox("Select case", txn_ids, index=default)
    case = next((c for c in cases if c["transaction_id"] == selected_id), None)

    if not case:
        st.warning("Case not found.")
        st.stop()

    rc        = case.get("risk_class", "LOW")
    score     = case.get("risk_score", 0)
    case_id   = case.get("case_id") or "—"
    ml_out    = case.get("ml_output", {})
    shap_data = ml_out.get("shap_explanation", {})
    top_feats = shap_data.get("top_features", [])
    llm_expl  = case.get("llm_explanation") or {}
    llm_judge = case.get("llm_judge") or {}

    # ── Header row ────────────────────────────────────────────────────────
    h1, h2, h3 = st.columns([3, 1.5, 1.5])
    with h1:
        st.subheader(f"Case: {selected_id}")
        st.caption(f"Case ID: {case_id}  ·  Model: {ml_out.get('model_version','?')}  ·  Mode: {ml_out.get('scorer_mode','?')}")
    with h2:
        st.markdown(f"**Risk Class:** {risk_badge(rc)}", unsafe_allow_html=False)
        st.markdown(risk_badge(rc))
    with h3:
        rec = llm_judge.get("recommendation", "—")
        st.markdown("**Judge Advisory:**")
        st.markdown(recommendation_badge(rec), unsafe_allow_html=True)

    st.divider()

    # ── Two-column layout ─────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        # Risk gauge
        st.plotly_chart(
            score_gauge(score, rc),
            use_container_width=True,
            key="gauge",
        )

        # SHAP waterfall
        if top_feats:
            st.plotly_chart(
                shap_waterfall(top_feats),
                use_container_width=True,
                key="shap",
            )
        else:
            st.info("No SHAP data available.")

    with right:
        # LLM explanation
        st.subheader("LLM Explanation")
        if llm_expl:
            expl_text = llm_expl.get("explanation_text", "")
            fallback  = llm_expl.get("fallback_used", True)
            if fallback:
                reason = llm_expl.get("fallback_reason", "UNKNOWN")
                if reason == "NO_API_KEY":
                    st.caption("⚠️ Fallback explanation — GEMINI_API_KEY not set")
                else:
                    st.caption(f"⚠️ Fallback explanation — {reason}")
            st.info(expl_text)

            codes = llm_expl.get("reason_codes", [])
            if codes:
                st.markdown("**Reason codes:** " + " · ".join(f"`{c}`" for c in codes))

            citations = llm_expl.get("policy_citations", [])
            if citations:
                st.markdown("**Policy citations:**")
                for cit in citations:
                    st.markdown(f"- **{cit.get('policy_id')}** — {cit.get('passage','')[:100]}…")
        else:
            st.info("LLM explanation not generated (LOW risk — skipped).")

        st.divider()

        # Judge output
        st.subheader("Judge Advisory")
        if llm_judge:
            judge_fallback = llm_judge.get("fallback_used", True)
            if judge_fallback:
                reason = llm_judge.get("fallback_reason", "UNKNOWN")
                if reason == "NO_API_KEY":
                    st.caption("⚠️ Heuristic advisory — GROQ_API_KEY not set")
                else:
                    st.caption(f"⚠️ Heuristic advisory — {reason}")

            conf = llm_judge.get("confidence", "LOW")
            st.markdown(f"**Recommendation:** {recommendation_badge(rec)} &nbsp; **Confidence:** `{conf}`",
                        unsafe_allow_html=True)
            st.markdown(f"_{llm_judge.get('narrative','')}_")

            signals = llm_judge.get("supporting_signals", [])
            if signals:
                st.markdown("**Supporting signals:**")
                for s in signals:
                    st.markdown(f"- {s}")
        else:
            st.info("Judge advisory not generated (LOW risk — skipped).")

    st.divider()

    # ── Reviewer actions ──────────────────────────────────────────────────
    st.subheader("Reviewer Decision")
    st.caption("All actions are permanently recorded in the audit trail.")

    ac1, ac2, ac3 = st.columns([2, 2, 3])
    with ac1:
        action = st.selectbox(
            "Action",
            ["APPROVE", "REJECT", "ESCALATE", "HOLD", "CLOSE"],
            key="reviewer_action",
        )
    with ac2:
        reason_codes = {
            "APPROVE": ["RISK_ACCEPTABLE", "FALSE_POSITIVE", "BUSINESS_JUSTIFIED"],
            "REJECT":  ["FALSE_POSITIVE", "INSUFFICIENT_EVIDENCE", "KNOWN_CUSTOMER"],
            "ESCALATE":["HIGH_RISK_CONFIRMED", "SANCTIONS_CONCERN", "SAR_WARRANTED"],
            "HOLD":    ["PENDING_INFO", "AWAITING_DOCUMENTS", "UNDER_INVESTIGATION"],
            "CLOSE":   ["RESOLVED", "NO_ACTION_REQUIRED", "DUPLICATE"],
        }
        reason = st.selectbox(
            "Reason code",
            reason_codes.get(action, ["OTHER"]),
            key="reviewer_reason",
        )
    with ac3:
        notes = st.text_input("Notes (optional)", key="reviewer_notes")

    if st.button("✅ Submit Decision", type="primary", use_container_width=True):
        event_record = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "event_type":     "REVIEWER_ACTION",
            "case_id":        case_id,
            "transaction_id": selected_id,
            "action":         action,
            "reason_code":    reason,
            "notes":          notes,
            "reviewer":       "analyst@deloitte.com",  # would come from auth
        }
        st.session_state.audit_events.insert(0, event_record)

        # Update case status in-memory
        for c in st.session_state.cases:
            if c["transaction_id"] == selected_id:
                c["_reviewer_action"] = action
                c["_reviewer_reason"] = reason
                break

        action_messages = {
            "APPROVE":  "✅ Case approved and closed.",
            "REJECT":   "❌ Alert rejected as false positive.",
            "ESCALATE": "🚨 Case escalated to senior compliance.",
            "HOLD":     "⏸ Case placed on hold.",
            "CLOSE":    "📁 Case closed with resolution.",
        }
        st.success(action_messages.get(action, "Action recorded."))
        st.balloons()


# ════════════════════════════════════════════════════════════════════════
# PAGE 3: AUDIT LOG
# ════════════════════════════════════════════════════════════════════════

elif page == "📋 Audit Log":
    st.title("Audit Log")
    st.caption("Immutable hash-chain event log — every pipeline step, MCP call, and reviewer action.")

    events = st.session_state.audit_events
    cases  = st.session_state.cases

    # Collect pipeline audit events from cases
    pipeline_events = []
    for c in cases:
        pipeline_events.append({
            "timestamp":      c.get("_scored_at", ""),
            "event_type":     "PIPELINE_COMPLETE",
            "transaction_id": c.get("transaction_id", ""),
            "risk_class":     c.get("risk_class", ""),
            "risk_score":     c.get("risk_score", 0),
            "case_id":        c.get("case_id", ""),
        })

    all_events = events + pipeline_events
    all_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    if not all_events:
        st.info("No audit events yet. Score a transaction to generate events.")
    else:
        st.markdown(f"**{len(all_events)} events** recorded")
        st.divider()

        for ev in all_events:
            ts    = ev.get("timestamp", "")[:19].replace("T", " ")
            etype = ev.get("event_type", "")

            icon = {
                "REVIEWER_ACTION":   "👤",
                "PIPELINE_COMPLETE": "⚙️",
                "CASE_CREATED":      "📂",
                "ML_SCORE_PRODUCED": "🤖",
                "LLM_OUTPUTS_PRODUCED": "💬",
            }.get(etype, "📝")

            with st.expander(f"{icon} `{ts}` — **{etype}**", expanded=False):
                display = {k: v for k, v in ev.items() if k != "timestamp"}
                st.json(display)
