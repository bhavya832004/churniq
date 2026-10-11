"""ChurnIQ Streamlit dashboard — reads artifacts produced by the CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from churniq.business import ASSUMPTION_NOTE, campaign_scenarios, sensitivity_analysis
from churniq.dashboard import load_csv, load_json, mask_customer_id, select_out_of_sample_scores

ARTIFACTS = Path("artifacts")
PIPELINE_CMD = (
    "python -m churniq.cli all --data data/raw/Telco-Customer-Churn.csv --out artifacts"
)


def _exists(*names: str) -> bool:
    return all((ARTIFACTS / name).exists() for name in names)


def _load_json(name: str) -> dict | None:
    return load_json(ARTIFACTS / name)


def _load_csv(name: str) -> pd.DataFrame | None:
    return load_csv(ARTIFACTS / name)


def _show_image(name: str, caption: str | None = None) -> None:
    path = ARTIFACTS / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Missing image artifact: `{name}`")


def _download_csv(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
    )


def _oos_scores() -> pd.DataFrame | None:
    oos = _load_csv("out_of_sample_risk_scores.csv")
    if oos is None:
        all_scores = _load_csv("customer_risk_scores.csv")
        if all_scores is None:
            return None
        if "score_method" not in all_scores.columns and "dataset_split" not in all_scores.columns:
            st.warning(
                "Score file has no split/method labels; treating all rows as display-only "
                "and not as unbiased predictions."
            )
        oos = select_out_of_sample_scores(all_scores)
    else:
        oos = select_out_of_sample_scores(oos)
    return oos


st.set_page_config(page_title="ChurnIQ", page_icon="📉", layout="wide")
st.title("ChurnIQ — Customer Retention Intelligence")
st.caption("Portfolio demo using IBM Telco Customer Churn (public teaching data).")

required_ok = _exists("test_metrics.json") and (
    _exists("customer_risk_scores.csv") or _exists("out_of_sample_risk_scores.csv")
)
if not required_ok:
    st.warning("Pipeline artifacts not found. Run this first:")
    st.code(PIPELINE_CMD)
    st.stop()

metrics = _load_json("test_metrics.json") or {}
quality = _load_json("data_quality.json") or {}
meta = _load_json("run_metadata.json") or {}
eda_summary = _load_json("eda_summary.json") or {}
oos = _oos_scores()
if oos is None or oos.empty:
    st.warning("No risk scores available. Re-run the pipeline.")
    st.code(PIPELINE_CMD)
    st.stop()

with st.sidebar:
    st.markdown("### Run metadata")
    st.write(
        {
            "selected_model": meta.get("selected_model") or metrics.get("selected_model"),
            "model_version": (meta.get("scoring") or {}).get("model_version"),
            "random_state": meta.get("random_state"),
            "split": meta.get("split"),
            "package_versions": meta.get("package_versions"),
        }
    )
    st.markdown("### Scenario inputs (illustrative)")
    value = st.number_input(
        "Assumed contribution value per retained customer", 0.0, 100000.0, 500.0, 25.0
    )
    cost = st.number_input("Assumed offer cost per targeted customer", 0.0, 10000.0, 25.0, 5.0)
    success = st.slider("Assumed offer success rate", 0.0, 1.0, 0.20, 0.05)
    cap = st.number_input("Max customers targeted (0 = no cap)", 0, int(len(oos)), 0, 10)
    budget = st.number_input("Campaign budget (0 = no budget cap)", 0.0, 1_000_000.0, 0.0, 100.0)

tabs = st.tabs(
    [
        "Executive summary",
        "Data quality",
        "Churn patterns",
        "Model evaluation",
        "Risk scores",
        "Campaign scenarios",
        "Methodology",
    ]
)

with tabs[0]:
    st.subheader("Executive summary")
    c1, c2, c3, c4 = st.columns(4)
    rows = quality.get("row_count") or eda_summary.get("customers") or metrics.get("train_rows")
    churn_rate = quality.get("churn_rate")
    if churn_rate is None:
        churn_rate = eda_summary.get("churn_rate")
    c1.metric("Customers (dataset)", rows if rows is not None else "—")
    c2.metric("Churn prevalence", f"{churn_rate:.1%}" if churn_rate is not None else "—")
    c3.metric("Selected model", metrics.get("selected_model", "—"))
    c4.metric("Test PR-AUC", f"{metrics['pr_auc']:.3f}" if metrics.get("pr_auc") is not None else "—")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall", f"{metrics['recall']:.3f}" if metrics.get("recall") is not None else "—")
    m2.metric("Precision", f"{metrics['precision']:.3f}" if metrics.get("precision") is not None else "—")
    m3.metric("F1", f"{metrics['f1']:.3f}" if metrics.get("f1") is not None else "—")
    m4.metric("Brier", f"{metrics['brier_score']:.3f}" if metrics.get("brier_score") is not None else "—")
    st.info(
        "Public teaching dataset only — not real customer advice. "
        "Campaign calculator uses assumed success rates, not causal uplift."
    )
    if meta.get("split"):
        st.write("Split sizes:", meta["split"])

with tabs[1]:
    st.subheader("Data quality")
    if not quality:
        st.info("Missing `data_quality.json`. Re-run audit/pipeline.")
    else:
        st.write(
            {
                "ok": quality.get("ok"),
                "source_path": quality.get("source_path"),
                "row_count": quality.get("row_count"),
                "churn_rate": quality.get("churn_rate"),
                "errors": quality.get("errors"),
                "warnings": quality.get("warnings"),
            }
        )
        metrics_block = quality.get("metrics") or {}
        missing = metrics_block.get("missing_values_by_column") or quality.get(
            "missing_values_by_column"
        )
        if missing:
            st.markdown("**Missing values**")
            st.dataframe(
                pd.DataFrame({"column": list(missing), "missing": list(missing.values())}),
                use_container_width=True,
            )
        notes = quality.get("preprocessing_notes") or []
        if notes:
            st.markdown("**Preprocessing notes**")
            for note in notes:
                st.write(f"- {note}")
        _download_csv("Download data quality metrics CSV", pd.json_normalize(quality), "data_quality.csv")

with tabs[2]:
    st.subheader("Churn patterns / EDA")
    st.caption("Observational associations only — not causal effects.")
    cols = st.columns(2)
    with cols[0]:
        _show_image("churn_distribution.png", "Churn prevalence")
        _show_image("tenure_vs_churn.png", "Tenure by churn")
    with cols[1]:
        _show_image("churn_rate_by_contract.png", "Churn rate by contract")
        _show_image("monthly_charges_by_churn.png", "Monthly charges by churn")

    segment_options = {
        "Contract": "churn_by_contract.csv",
        "Tenure band": "churn_by_tenure.csv",
        "Internet service": "churn_by_internet.csv",
        "Payment method": "churn_by_payment.csv",
        "Paperless billing": "churn_by_paperless.csv",
    }
    available = {label: name for label, name in segment_options.items() if _exists(name)}
    if available:
        choice = st.selectbox("Segment table", list(available))
        segment = _load_csv(available[choice])
        if segment is not None:
            st.dataframe(segment, use_container_width=True)
            if "churn_rate" in segment.columns and "customers" in segment.columns:
                chart_df = segment.copy()
                label_col = [c for c in chart_df.columns if c not in {
                    "customers", "churners", "churn_rate", "ci_low_95", "ci_high_95"
                }]
                if label_col:
                    chart_df = chart_df.set_index(label_col[0])[["churn_rate"]]
                    st.bar_chart(chart_df)
            _download_csv(f"Download {choice} CSV", segment, available[choice])
    else:
        st.info("No segment EDA CSVs found.")

with tabs[3]:
    st.subheader("Model evaluation")
    st.caption(metrics.get("evaluation_protocol") or "Test metrics are for the selected model only.")
    comparison = _load_csv("model_comparison.csv")
    if comparison is not None:
        st.markdown("**Validation comparison (selection metrics)**")
        st.dataframe(comparison, use_container_width=True)
        _download_csv("Download model comparison CSV", comparison, "model_comparison.csv")
    report = _load_csv("classification_report.csv")
    if report is not None:
        st.markdown("**Per-class report (test)**")
        st.dataframe(report, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        _show_image("confusion_matrix.png", "Test confusion matrix")
    with c2:
        _show_image("calibration_curve.png", "Calibration curve")
    lift = _load_csv("lift_by_decile.csv")
    if lift is not None:
        st.markdown("**Lift / gains by risk decile (test)**")
        st.dataframe(lift, use_container_width=True)
        if "lift_vs_average" in lift.columns:
            st.bar_chart(lift.set_index("risk_decile")[["lift_vs_average"]])
        _download_csv("Download lift CSV", lift, "lift_by_decile.csv")
    errors = _load_csv("error_analysis.csv")
    if errors is not None:
        st.markdown("**Error analysis (test misclassifications)**")
        view = errors.copy()
        if "customerID" in view.columns:
            view["customerID"] = mask_customer_id(view["customerID"])
        st.dataframe(view.head(100), use_container_width=True)

with tabs[4]:
    st.subheader("Risk distribution and scores")
    st.caption(
        "Out-of-sample scores only in this view. Train in-sample scores are excluded from rankings."
    )
    if "risk_tier" in oos.columns:
        st.bar_chart(
            oos["risk_tier"]
            .value_counts()
            .reindex(["Very high", "High", "Moderate", "Low"], fill_value=0)
        )
    if "dataset_split" in oos.columns:
        splits = sorted(oos["dataset_split"].dropna().unique().tolist())
        selected_splits = st.multiselect("Filter dataset_split", splits, default=splits)
        view = oos.loc[oos["dataset_split"].isin(selected_splits)].copy()
    else:
        view = oos.copy()
    if "risk_tier" in view.columns:
        tiers = st.multiselect(
            "Filter risk_tier",
            ["Very high", "High", "Moderate", "Low"],
            default=["Very high", "High", "Moderate", "Low"],
        )
        view = view.loc[view["risk_tier"].isin(tiers)]
    display = view.sort_values("churn_probability", ascending=False).head(200).copy()
    if "customerID" in display.columns:
        display["customerID"] = mask_customer_id(display["customerID"])
    # Keep export columns limited.
    keep = [
        c
        for c in [
            "customerID",
            "dataset_split",
            "score_method",
            "churn_probability",
            "risk_tier",
            "threshold_decision",
            "model_name",
            "model_version",
            "scored_at_utc",
        ]
        if c in display.columns
    ]
    st.dataframe(display[keep] if keep else display, use_container_width=True)
    _download_csv("Download out-of-sample risk scores CSV", oos, "out_of_sample_risk_scores.csv")

with tabs[5]:
    st.subheader("Retention campaign scenarios")
    st.warning(ASSUMPTION_NOTE)
    st.caption("All monetary defaults are illustrative and editable in the sidebar.")
    scenario = campaign_scenarios(
        oos["churn_probability"],
        customer_value=value,
        offer_cost=cost,
        offer_success_rate=success,
        max_customers=int(cap) if cap else None,
        budget=float(budget) if budget else None,
    )
    st.dataframe(scenario, use_container_width=True)
    _download_csv("Download scenario table CSV", scenario, "campaign_scenarios_live.csv")
    sensitivity = sensitivity_analysis(
        oos["churn_probability"],
        customer_value=value,
        offer_cost=cost,
        offer_success_rate=success,
        max_customers=int(cap) if cap else None,
        budget=float(budget) if budget else None,
    )
    st.markdown("**Sensitivity to assumptions**")
    st.dataframe(sensitivity, use_container_width=True)
    success_sens = sensitivity.loc[
        sensitivity["varied_parameter"] == "assumed_offer_success_rate",
        ["varied_value", "scenario_net_value"],
    ].set_index("varied_value")
    if not success_sens.empty:
        st.line_chart(success_sens)
    _download_csv("Download sensitivity CSV", sensitivity, "campaign_sensitivity_live.csv")

with tabs[6]:
    st.subheader("Methodology and limitations")
    st.markdown(
        """
- Target is churn classification on a static teaching snapshot, not time-horizon forecasting.
- Preprocessing fits on training data inside sklearn pipelines.
- Model selected by **validation PR-AUC**; threshold by **validation F2**; test evaluated once.
- Risk scores are labeled by split/method; train in-sample scores are not unbiased predictions.
- Campaign outputs are **scenario assumptions**, not causal uplift / individual treatment effects.
- Dataset is public teaching data — not advice for real customers.
        """
    )
    if metrics.get("accuracy_note"):
        st.info(metrics["accuracy_note"])
    if meta:
        st.markdown("**Run metadata**")
        st.json(
            {
                "selected_model": meta.get("selected_model"),
                "selection": meta.get("selection"),
                "scoring": meta.get("scoring"),
                "feature_columns": meta.get("feature_columns"),
                "note": meta.get("note"),
            }
        )
    st.code(PIPELINE_CMD)
