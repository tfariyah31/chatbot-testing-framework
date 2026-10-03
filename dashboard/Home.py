# dashboard/Home.py
# ConvoQA Dashboard — Home/Overview page

import streamlit as st
import json
import os

# ─── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="ConvoQA — AI Testing Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Data ────────────────────────────────────────────────

@st.cache_data
def load_results():
    path = os.path.join(
        os.path.dirname(__file__),
        "data/test_results.json"
    )
    with open(path) as f:
        return json.load(f)

data = load_results()

# ─── Header ───────────────────────────────────────────────────

st.title("🤖 ConvoQA — Conversational AI Testing Framework")
st.markdown(
    "An automated framework for evaluating conversational AI "
    "systems across functional, adversarial, and regression "
    "testing dimensions."
)
st.divider()

# ─── Key Stats Row ────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Tests",
        value=data["framework"]["total_tests"]
    )
with col2:
    st.metric(
        label="Overall Pass Rate",
        value=f"{data['summary']['overall_pass_rate']}%",
        delta="All suites green"
    )
with col3:
    st.metric(
        label="Critical Bugs Found",
        value=data["summary"]["critical_bugs_found"],
        delta="All fixed",
        delta_color="inverse"
    )
with col4:
    st.metric(
        label="Prompt Version",
        value=f"v{data['framework']['prompt_version']}"
    )
with col5:
    st.metric(
        label="Evaluation Model",
        value="GPT-4o-mini"
    )

st.divider()

# ─── Suite Results ────────────────────────────────────────────

st.subheader("Test Suite Results")

col1, col2, col3 = st.columns(3)

with col1:
    rate = data["summary"]["functional_pass_rate"]
    st.metric("Functional Tests", f"{rate}%")
    st.progress(rate / 100)
    st.caption("Intent recognition, semantic evaluation, "
               "context retention")

with col2:
    rate = data["summary"]["adversarial_pass_rate"]
    st.metric("Adversarial Tests", f"{rate}%")
    st.progress(rate / 100)
    st.caption("Prompt injection, jailbreaking, "
               "social engineering")

with col3:
    rate = data["summary"]["regression_pass_rate"]
    st.metric("Regression Tests", f"{rate}%")
    st.progress(rate / 100)
    st.caption("Capability, safety, and persona "
               "stability across versions")

st.divider()

# ─── Tech Stack ───────────────────────────────────────────────

st.subheader("Tech Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Testing & Evaluation**
    - Pytest — test runner and suite organisation
    - DeepEval — LLM evaluation framework
    - GEval — custom plain-English evaluation criteria
    - GPT-4o-mini — evaluator model
    """)

with col2:
    st.markdown("""
    **Bot Under Test**
    - NutriBot — nutrition and recipe assistant
    - GPT-4o-mini — underlying model
    - Custom system prompt — v1.2 (hardened)
    - 3 prompt versions tracked with regression suite
    """)

st.divider()

# ─── Critical Findings ────────────────────────────────────────

st.subheader("Critical Security Findings")

for bug in data["bugs"]:
    severity_color = "🔴" if bug["severity"] == "CRITICAL" else "🟡"
    status_icon = "✅" if bug["status"] == "Fixed" else "🔧"

    with st.expander(
        f"{severity_color} {bug['id']} — {bug['title']}"
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Category:** {bug['category']}")
            st.markdown(f"**Severity:** {bug['severity']}")
        with col2:
            st.markdown(f"**Found in:** v{bug['found_in_version']}")
            st.markdown(f"**Fixed in:** v{bug['fixed_in_version']}")
        with col3:
            st.markdown(f"**Status:** {status_icon} {bug['status']}")

st.divider()

# ─── About ────────────────────────────────────────────────────

st.subheader("About This Project")
st.markdown("""
Built by **Fariyah** — Lead QA Manager with 14 years of QA/SDET 
experience, pivoting to AI Evaluation Engineering.

This framework demonstrates the application of traditional QA 
rigour to conversational AI systems — combining structured test 
design, semantic evaluation, adversarial testing, and regression 
management into a professional portfolio project.

**GitHub:** github.com/tfariyah31
""")