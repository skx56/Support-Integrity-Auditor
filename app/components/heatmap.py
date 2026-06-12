"""
app/components/heatmap.py
Severity Delta Heatmap: rows=Ticket Type, cols=Channel, color=avg severity delta.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st


def render_severity_heatmap(df: pd.DataFrame):
    """
    Renders a heatmap of average severity_delta across
    ticket_type (rows) × ticket_channel (columns).
    """
    required = ["ticket_type", "ticket_channel", "severity_delta"]
    if not all(c in df.columns for c in required):
        st.info("Heatmap unavailable: missing required columns.")
        return

    pivot = (
        df.groupby(["ticket_type", "ticket_channel"])["severity_delta"]
          .mean()
          .unstack(fill_value=0)
    )

    types = pivot.index.tolist()
    channels = pivot.columns.tolist()
    z = pivot.values

    # Custom diverging colorscale: red (crisis) → white (neutral) → green (false alarm)
    colorscale = [
        [0.0,  "#dc2626"],  # Critical under-prioritised (red)
        [0.35, "#fca5a5"],
        [0.5,  "#1e293b"],  # Neutral (dark)
        [0.65, "#86efac"],
        [1.0,  "#16a34a"],  # Over-prioritised (green)
    ]

    text = np.vectorize(lambda v: f"{v:+.2f}")(z)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=channels,
        y=types,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=colorscale,
        zmid=0,
        hovertemplate=(
            "<b>Type:</b> %{y}<br>"
            "<b>Channel:</b> %{x}<br>"
            "<b>Avg Severity Delta:</b> %{z:+.2f}<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Avg Δ Severity", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0"),
            tickformat="+.1f",
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Severity Delta Heatmap (Ticket Type × Channel)",
            font=dict(color="#e2e8f0", size=16),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Channel",
            color="#94a3b8",
            tickangle=-30,
        ),
        yaxis=dict(
            title="Ticket Type",
            color="#94a3b8",
            autorange="reversed",
        ),
        margin=dict(t=60, b=60, l=160, r=40),
        height=max(300, 60 * len(types) + 100),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend explanation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔴 **Red** = Hidden Crisis zone  \nTickets severely undervalued")
    with col2:
        st.markdown("⚫ **Dark** = Consistent zone  \nPriority matches severity")
    with col3:
        st.markdown("🟢 **Green** = False Alarm zone  \nTickets overvalued")


def render_confidence_histogram(df: pd.DataFrame):
    """Histogram of model confidence scores."""
    if "mismatch_confidence" not in df.columns:
        return

    import plotly.express as px
    fig = px.histogram(
        df,
        x="mismatch_confidence",
        color="mismatch_type" if "mismatch_type" in df.columns else None,
        nbins=30,
        title="Model Confidence Distribution",
        labels={"mismatch_confidence": "Mismatch Probability", "count": "# Tickets"},
        color_discrete_map={
            "Consistent":    "#22c55e",
            "Hidden Crisis": "#ef4444",
            "False Alarm":   "#f97316",
        },
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0")),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
