"""Streamlit interface for HealthPlan Intelligence."""

from __future__ import annotations

import streamlit as st

from healthplan_intelligence import (
    explain_prediction,
    generate_demo_data,
    predict_high_cost_risk,
    result_summary,
    train_model,
)


st.set_page_config(
    page_title="HealthPlan Intelligence",
    page_icon="🫶",
    layout="centered",
)


@st.cache_resource
def prepare_demo():
    """Train once, then reuse the model while the app is open."""

    demo_data = generate_demo_data()
    model, metrics, benchmarks = train_model(demo_data)
    return demo_data, model, metrics, benchmarks


data, model, metrics, benchmarks = prepare_demo()

st.title("HealthPlan Intelligence")
st.subheader("Understand a health plan in about 30 seconds")
st.write(
    "Enter a few plan details. A small machine-learning model estimates cost "
    "exposure and explains the result in everyday language."
)

with st.form("plan_form"):
    plan_type = st.selectbox("Plan level", ["Bronze", "Silver", "Gold", "Platinum"])
    deductible = st.slider("Annual deductible", 0, 9_500, 4_000, step=100)
    primary_care_copay = st.slider("Primary-care copay", 0, 100, 35, step=5)
    coinsurance_percent = st.slider("Coinsurance after deductible", 0, 50, 20, step=5)
    annual_premium = st.slider("Annual premium", 2_000, 13_000, 5_400, step=100)
    hsa_eligible = st.checkbox("HSA eligible")
    submitted = st.form_submit_button("Help me understand this plan", use_container_width=True)

if submitted:
    plan = {
        "plan_type": plan_type,
        "deductible": deductible,
        "primary_care_copay": primary_care_copay,
        "coinsurance_percent": coinsurance_percent,
        "annual_premium": annual_premium,
        "hsa_eligible": hsa_eligible,
    }
    probability = predict_high_cost_risk(model, plan)
    label, tone = result_summary(probability)

    if tone == "high":
        st.error(f"{label} — {probability:.0%} model estimate")
    elif tone == "medium":
        st.warning(f"{label} — {probability:.0%} model estimate")
    else:
        st.success(f"{label} — {probability:.0%} model estimate")

    st.markdown("#### Why the model responded this way")
    for reason in explain_prediction(plan, probability, benchmarks):
        st.write(f"• {reason}")

with st.expander("See how this demo works"):
    st.write(
        "The app trains a random-forest classifier on fictional plan examples. "
        "It keeps preprocessing and prediction in one repeatable pipeline."
    )
    left, right = st.columns(2)
    left.metric("Test accuracy", f"{metrics['accuracy']:.0%}")
    right.metric("Test ROC-AUC", f"{metrics['roc_auc']:.2f}")
    st.dataframe(data.head(8), use_container_width=True, hide_index=True)

st.caption(
    "Educational portfolio demo using synthetic data. It is not insurance, "
    "medical, legal, or financial advice. Always review official plan documents."
)
