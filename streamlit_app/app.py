# -*- coding: utf-8 -*-
"""Streamlit app for the Best Time To Call predictive analytics project."""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "btc_connect_probability_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
RECOMMENDATION_PATH = PROJECT_ROOT / "outputs" / "future_best_time_recommendations.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "feature_importance.csv"


def time_class(hour: int) -> str:
    if 8 <= hour <= 10:
        return "Morning"
    if 11 <= hour <= 13:
        return "Noon"
    if 14 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 19:
        return "Evening"
    return "Outside Business Hours"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_recommendations():
    return pd.read_csv(RECOMMENDATION_PATH)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH).head(20)


st.set_page_config(page_title="Best Time To Call Predictor", page_icon="☎️", layout="wide")
st.title("☎️ Best Time To Call Predictor")
st.caption("Synthetic-data portfolio project: outbound-call connection probability and recommended call windows.")

if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run `python src/train_model.py` before launching the app.")
    st.stop()

model = load_model()
metrics = load_metrics()
recommendations = load_recommendations()
feature_importance = load_feature_importance()

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", metrics.get("roc_auc"))
col2.metric("PR-AUC", metrics.get("pr_auc"))
col3.metric("F1 Score", metrics.get("f1"))
col4.metric("Decision Threshold", metrics.get("threshold"))

st.markdown("---")
st.subheader("Predict a Call Window")

left, right = st.columns([1, 2])
with left:
    call_date = st.date_input("Planned call date", value=pd.to_datetime("2026-01-05"))
    hour = st.slider("Call hour", 8, 19, 10)
    minute = st.selectbox("Call minute", [0, 15, 30, 45], index=0)
    agent_name = st.selectbox(
        "Agent group",
        ["t1.tlp.tso.ms.agent_01", "t1.tlp.tso.ms.agent_12", "external.agent_01"],
    )
    attempt_number = st.slider("Caller attempt number", 1, 8, 1)

attempt_bucket = "1" if attempt_number == 1 else "2" if attempt_number == 2 else "3" if attempt_number == 3 else "4-5" if attempt_number <= 5 else "6+"
input_df = pd.DataFrame(
    [
        {
            "Original Call Hour": hour,
            "Original Call Minute": minute,
            "Month": pd.Timestamp(call_date).month,
            "DayOfWeek": pd.Timestamp(call_date).weekday(),
            "IsWeekend": int(pd.Timestamp(call_date).weekday() >= 5),
            "WeekName": pd.Timestamp(call_date).day_name(),
            "MonthName": pd.Timestamp(call_date).month_name(),
            "TimeClass": time_class(hour),
            "Agent Name": agent_name,
            "CallerAttemptNumber": attempt_number,
            "CallerAttemptBucket": attempt_bucket,
        }
    ]
)
probability = model.predict_proba(input_df)[0, 1]
label = "Recommended" if probability >= metrics.get("threshold", 0.5) else "Lower Priority"

with right:
    st.metric("Predicted connect probability", f"{probability:.1%}")
    st.write(f"Recommendation: **{label}**")
    st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.subheader("Top Recommended Future Call Windows")
filter_day = st.multiselect("Filter day", sorted(recommendations["WeekName"].unique()), default=list(sorted(recommendations["WeekName"].unique()))[:3])
filtered = recommendations[recommendations["WeekName"].isin(filter_day)].head(25)
st.dataframe(filtered, use_container_width=True)

chart_df = (
    recommendations.groupby(["WeekName", "Original Call Hour"], as_index=False)["Predicted_Connect_Probability"].mean()
)
st.bar_chart(chart_df.pivot(index="Original Call Hour", columns="WeekName", values="Predicted_Connect_Probability"))

st.markdown("---")
st.subheader("Top Model Drivers")
st.dataframe(feature_importance, use_container_width=True)
st.bar_chart(feature_importance.set_index("feature")["importance"])

st.info(
    "Portfolio note: this app uses synthetic data only. The original client dataset was replaced to protect confidentiality while preserving the analytics problem, fields, and business logic."
)
