"""
app/components/dossier_viewer.py
Evidence Dossier display component for Streamlit.
Renders a structured, visually rich dossier card.
"""
import streamlit as st
import json
from typing import Dict, Any


MISMATCH_COLORS = {
    "Hidden Crisis": ("🔴", "#ef4444", "#450a0a"),
    "False Alarm":   ("🟠", "#f97316", "#431407"),
    "Consistent":    ("✅", "#22c55e", "#052e16"),
}

SIGNAL_ICONS = {
    "keyword":       "🔤",
    "resolution_time": "⏱️",
    "llm_severity":  "🧠",
    "channel":       "📡",
    "customer_tier": "🏢",
    "embedding":     "🔵",
}


def render_dossier_card(dossier: Dict[str, Any]):
    """
    Renders a full Evidence Dossier as a styled Streamlit card.
    """
    mtype = dossier.get("mismatch_type", "Hidden Crisis")
    icon, color, bg_color = MISMATCH_COLORS.get(mtype, ("⚠️", "#f59e0b", "#1c1917"))

    confidence = float(dossier.get("confidence", 0))
    delta = dossier.get("severity_delta", "+0")

    # Header
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {bg_color} 0%, #1e293b 100%);
        border: 1px solid {color};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 24px;">{icon}</span>
                <span style="color: {color}; font-size: 20px; font-weight: 700; margin-left: 8px;">
                    {mtype}
                </span>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 12px;">Ticket ID</div>
                <div style="color: #e2e8f0; font-weight: 600;">{dossier.get("ticket_id", "?")}</div>
            </div>
        </div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div>
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Assigned</div>
                <div style="color: #e2e8f0; font-size: 18px; font-weight: 600;">{dossier.get("assigned_priority", "?")}</div>
            </div>
            <div style="font-size: 24px; color: #475569; align-self: center;">→</div>
            <div>
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Inferred</div>
                <div style="color: {color}; font-size: 18px; font-weight: 600;">{dossier.get("inferred_severity", "?")}</div>
            </div>
            <div style="margin-left: auto; text-align: right;">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Severity Delta</div>
                <div style="color: {color}; font-size: 22px; font-weight: 700;">{delta}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Confidence</div>
                <div style="color: #e2e8f0; font-size: 18px; font-weight: 600;">{confidence:.0%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Evidence items
    st.markdown("**Evidence Chain**")
    for ev in dossier.get("feature_evidence", []):
        signal = ev.get("signal", "unknown")
        icon_ev = SIGNAL_ICONS.get(signal, "📌")
        weight = ev.get("weight", "?")

        with st.expander(f"{icon_ev} {signal.replace('_', ' ').title()} — weight {weight}", expanded=True):
            st.markdown(f"**Field**: `{ev.get('field', signal)}`")
            st.markdown(f"**Value**: {ev.get('value', '')}")
            if "interpretation" in ev:
                st.markdown(f"**Interpretation**: *{ev['interpretation']}*")
            if "all_matched" in ev and ev["all_matched"]:
                st.markdown(f"**All matched keywords**: `{', '.join(ev['all_matched'])}`")
            if ev.get("grounded") is False:
                st.warning("⚠️ LLM reason could not be fully verified against source text.")

    # Constraint analysis
    st.markdown("**Constraint Analysis**")
    st.markdown(f"""
    <div style="
        background: #1e293b;
        border-left: 4px solid {color};
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        color: #cbd5e1;
        font-style: italic;
        line-height: 1.6;
    ">
    {dossier.get("constraint_analysis", "")}
    </div>
    """, unsafe_allow_html=True)

    # Raw JSON expander
    with st.expander("📄 Raw Dossier JSON"):
        st.code(json.dumps(dossier, indent=2), language="json")


def render_dossier_table(dossiers: list):
    """Compact table summary of all dossiers."""
    import pandas as pd

    if not dossiers:
        st.info("No mismatched tickets found.")
        return

    rows = []
    for d in dossiers:
        rows.append({
            "Ticket ID":         d.get("ticket_id", "?"),
            "Type":              d.get("mismatch_type", "?"),
            "Assigned":          d.get("assigned_priority", "?"),
            "Inferred":          d.get("inferred_severity", "?"),
            "Delta":             d.get("severity_delta", "?"),
            "Confidence":        f"{float(d.get('confidence', 0)):.0%}",
            "Top Signal":        d["feature_evidence"][0].get("signal", "?") if d.get("feature_evidence") else "?",
        })

    df = pd.DataFrame(rows)

    def color_type(val):
        if val == "Hidden Crisis":
            return "color: #ef4444; font-weight: bold"
        elif val == "False Alarm":
            return "color: #f97316; font-weight: bold"
        return ""

    styled = df.style.applymap(color_type, subset=["Type"])
    st.dataframe(styled, use_container_width=True, height=300)
