from pathlib import Path
import json
import pandas as pd
import streamlit as st
from churniq.business import campaign_scenarios

st.set_page_config(page_title="ChurnIQ", page_icon="📉", layout="wide")
st.title("ChurnIQ — Customer Retention Intelligence")
st.caption("Portfolio demo using IBM Telco Customer Churn.")
out = Path("artifacts")
if not (out / "test_metrics.json").exists() or not (out / "customer_risk_scores.csv").exists():
    st.warning("Run the pipeline first:")
    st.code("python -m churniq.cli all --data data/raw/Telco-Customer-Churn.csv --out artifacts")
    st.stop()
metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
scores = pd.read_csv(out / "customer_risk_scores.csv")
a,b,c,d = st.columns(4)
a.metric("Selected model", metrics["selected_model"])
b.metric("PR-AUC", f'{metrics["pr_auc"]:.3f}')
c.metric("Recall", f'{metrics["recall"]:.3f}')
d.metric("Precision", f'{metrics["precision"]:.3f}')
st.subheader("Risk tier distribution")
st.bar_chart(scores["risk_tier"].value_counts().reindex(["Very high","High","Moderate","Low"], fill_value=0))
if (out / "model_comparison.csv").exists():
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(out / "model_comparison.csv"), use_container_width=True)
st.subheader("Retention campaign scenario calculator")
st.caption("Assumptions are illustrative; this is not causal uplift modeling.")
with st.sidebar:
    value = st.number_input("Estimated contribution value per retained customer", 0.0, 100000.0, 500.0, 25.0)
    cost = st.number_input("Offer cost per targeted customer", 0.0, 10000.0, 25.0, 5.0)
    success = st.slider("Assumed offer success rate", 0.0, 1.0, 0.20, 0.05)
    cap = st.number_input("Max customers targeted (0 = no cap)", 0, int(len(scores)), 0, 10)
scenario = campaign_scenarios(scores["churn_probability"], value, cost, success, int(cap) if cap else None)
st.dataframe(scenario, use_container_width=True)
st.subheader("Highest-risk customers")
st.dataframe(scores.sort_values("churn_probability", ascending=False).head(100), use_container_width=True)
st.download_button("Download risk scores CSV", scores.to_csv(index=False).encode("utf-8"),
                   "customer_risk_scores.csv", "text/csv")
