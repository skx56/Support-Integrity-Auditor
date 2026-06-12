"""
app/components/dashboard.py
Priority Mismatch Dashboard components: distribution charts,
mismatch type breakdown, top signal contributions.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st


COLORS = {
    "consistent":    "#22c55e",  # green
    "hidden_crisis": "#ef4444",  # red
    "false_alarm":   "#f97316",  # orange
    "accent":        "#6366f1",  # indigo
    "bg":            "#0f172a",
    "card":          "#1e293b",
    "text":          "#e2e8f0",
}


def render_kpi_cards(df: pd.DataFrame):
    """Top KPI cards row."""
    total = len(df)
    mismatches = int(df["mismatch"].sum()) if "mismatch" in df.columns else 0
    hidden_crisis = int((df.get("mismatch_type", pd.Series()) == "Hidden Crisis").sum())
    false_alarm = int((df.get("mismatch_type", pd.Series()) == "False Alarm").sum())
    mismatch_rate = mismatches / total if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Tickets", f"{total:,}")
    with col2:
        st.metric("Mismatched", f"{mismatches:,}", delta=f"{mismatch_rate:.1%} of total")
    with col3:
        st.metric("🔴 Hidden Crisis", f"{hidden_crisis:,}",
                  help="Tickets undervalued — true severity exceeds assigned priority")
    with col4:
        st.metric("🟠 False Alarm", f"{false_alarm:,}",
                  help="Tickets overvalued — assigned priority exceeds true severity")
    with col5:
        consistent = total - mismatches
        st.metric("✅ Consistent", f"{consistent:,}")


def render_mismatch_donut(df: pd.DataFrame):
    """Donut chart: Consistent vs Hidden Crisis vs False Alarm."""
    if "mismatch_type" not in df.columns:
        return

    counts = df["mismatch_type"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()

    color_map = {
        "Consistent":    COLORS["consistent"],
        "Hidden Crisis": COLORS["hidden_crisis"],
        "False Alarm":   COLORS["false_alarm"],
    }
    colors = [color_map.get(l, "#94a3b8") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e2e8f0", size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(df):,}</b><br>tickets",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#e2e8f0"),
    )
    fig.update_layout(
        title=dict(text="Ticket Audit Distribution", font=dict(color="#e2e8f0", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_mismatch_by_channel(df: pd.DataFrame):
    """Stacked bar: mismatch type breakdown by channel."""
    if "ticket_channel" not in df.columns or "mismatch_type" not in df.columns:
        return

    pivot = (
        df.groupby(["ticket_channel", "mismatch_type"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    fig = go.Figure()
    color_map = {
        "Consistent":    COLORS["consistent"],
        "Hidden Crisis": COLORS["hidden_crisis"],
        "False Alarm":   COLORS["false_alarm"],
    }

    for col in ["Consistent", "Hidden Crisis", "False Alarm"]:
        if col in pivot.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=pivot["ticket_channel"],
                y=pivot[col],
                marker_color=color_map[col],
                hovertemplate=f"<b>{col}</b><br>Channel: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Mismatch Type by Channel", font=dict(color="#e2e8f0", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Channel", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(title="Ticket Count", color="#94a3b8", gridcolor="#334155"),
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(t=50, b=40, l=40, r=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_signals(df: pd.DataFrame):
    """Bar chart: signal contribution to mismatch detection."""
    if not all(c in df.columns for c in
               ["severity_llm", "severity_resolution", "severity_rules", "severity_cluster"]):
        return

    mismatch_df = df[df.get("mismatch", pd.Series(0)) == 1]
    if len(mismatch_df) == 0:
        return

    # Compute how often each signal agreed with final mismatch label
    signal_data = {
        "LLM Scoring (Phi-3-mini)": float((
            (mismatch_df["severity_llm"] - mismatch_df["priority_numeric"]).abs() >= 2
        ).mean()),
        "Resolution Time Regression": float((
            (mismatch_df["severity_resolution"] - mismatch_df["priority_numeric"]).abs() >= 2
        ).mean()),
        "Rule-Based NLP": float((
            (mismatch_df["severity_rules"] - mismatch_df["priority_numeric"]).abs() >= 2
        ).mean()),
        "Embedding Clustering": float((
            (mismatch_df["severity_cluster"] - mismatch_df["priority_numeric"]).abs() >= 2
        ).mean()),
    }

    signals = list(signal_data.keys())
    values = [v * 100 for v in signal_data.values()]

    fig = go.Figure(go.Bar(
        x=values,
        y=signals,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#334155"], [1, COLORS["accent"]]],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
        hovertemplate="<b>%{y}</b><br>Agreement: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Signal Agreement with Mismatch Label", font=dict(color="#e2e8f0", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="% Tickets Flagged by Signal", color="#94a3b8",
                   gridcolor="#334155", range=[0, 110]),
        yaxis=dict(color="#94a3b8"),
        margin=dict(t=50, b=40, l=200, r=60),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_priority_distribution(df: pd.DataFrame):
    """Grouped bar: assigned vs inferred priority distribution."""
    if "ticket_priority" not in df.columns or "inferred_severity_label" not in df.columns:
        return

    levels = ["Low", "Medium", "High", "Critical"]
    assigned_counts = df["ticket_priority"].value_counts().reindex(levels, fill_value=0)
    inferred_counts = df["inferred_severity_label"].value_counts().reindex(levels, fill_value=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Assigned Priority",
        x=levels,
        y=assigned_counts.values,
        marker_color="#6366f1",
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Inferred Severity",
        x=levels,
        y=inferred_counts.values,
        marker_color="#22d3ee",
        opacity=0.85,
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Assigned Priority vs. Inferred Severity", font=dict(color="#e2e8f0", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(title="Count", color="#94a3b8", gridcolor="#334155"),
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(t=50, b=40, l=40, r=20),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
