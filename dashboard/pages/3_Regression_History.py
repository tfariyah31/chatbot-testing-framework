# dashboard/pages/3_Regression_History.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pandas as pd

st.set_page_config(
    page_title="Regression History — ConvoQA",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_results():
    path = os.path.join(
        os.path.dirname(__file__),
        "../data/test_results.json"
    )
    with open(path) as f:
        return json.load(f)

data = load_results()
history = data["regression_history"]

# ─── Header ───────────────────────────────────────────────────

st.title("📈 Regression History")
st.markdown(
    "Tracking NutriBot quality across prompt versions. "
    "Every prompt change triggers a full regression run "
    "across capability, safety, and persona layers."
)
st.divider()

# ─── Version Timeline ─────────────────────────────────────────

st.subheader("Prompt Version Timeline")

df = pd.DataFrame(history)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["version"],
    y=df["capability_pass_rate"],
    name="Capability",
    mode="lines+markers",
    line=dict(color="rgb(99, 110, 250)", width=2),
    marker=dict(size=10)
))

fig.add_trace(go.Scatter(
    x=df["version"],
    y=df["safety_pass_rate"],
    name="Safety",
    mode="lines+markers",
    line=dict(color="rgb(239, 85, 59)", width=2),
    marker=dict(size=10)
))

fig.add_trace(go.Scatter(
    x=df["version"],
    y=df["persona_pass_rate"],
    name="Persona",
    mode="lines+markers",
    line=dict(color="rgb(0, 204, 150)", width=2),
    marker=dict(size=10)
))

fig.add_hline(
    y=90,
    line_dash="dash",
    line_color="gray",
    annotation_text="90% quality gate"
)

fig.update_layout(
    title="Pass Rate by Layer Across Prompt Versions",
    xaxis_title="Prompt Version",
    yaxis_title="Pass Rate (%)",
    yaxis_range=[50, 105],
    height=400,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Version Detail Cards ─────────────────────────────────────

st.subheader("Version Details")

for version in reversed(history):
    is_current = (
        version["version"] == data["framework"]["prompt_version"]
    )

    label = (
        f"v{version['version']} — {version['date']}"
        f"{' (current)' if is_current else ''}"
    )

    with st.expander(label, expanded=is_current):
        st.markdown(f"**Changes:** {version['changes']}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rate = version["capability_pass_rate"]
            st.metric(
                "Capability",
                f"{rate}%",
                delta="✅" if rate == 100 else "⚠️"
            )
        with col2:
            rate = version["safety_pass_rate"]
            st.metric(
                "Safety",
                f"{rate}%",
                delta="✅" if rate >= 90 else "⚠️"
            )
        with col3:
            rate = version["persona_pass_rate"]
            st.metric(
                "Persona",
                f"{rate}%",
                delta="✅" if rate == 100 else "⚠️"
            )
        with col4:
            bugs = version["bugs_found"]
            st.metric(
                "Bugs Found",
                bugs,
                delta=f"{bugs} fixed" if bugs > 0 else "None",
                delta_color="inverse" if bugs > 0 else "normal"
            )

st.divider()

# ─── Regression Methodology ───────────────────────────────────

st.subheader("Regression Testing Methodology")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Layer 1 — Capability**")
    st.markdown("""
    Verifies core functions remain intact:
    - Recipe generation
    - Dietary restriction handling
    - Nutritional information
    """)

with col2:
    st.markdown("**Layer 2 — Safety**")
    st.markdown("""
    Verifies guardrails remain strong:
    - Medical diagnosis refusal
    - Extreme diet blocking
    - Prompt injection resistance
    - System prompt confidentiality
    """)

with col3:
    st.markdown("**Layer 3 — Persona**")
    st.markdown("""
    Verifies character consistency:
    - Off-topic redirection
    - Tone under pressure
    - Identity stability
    """)