"""Train/validation/test modeling protocol for churn classification."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import (
    choose_threshold,
    choose_threshold_cost,
    compute_classification_metrics,
    lift_gains_table,
)
from .scoring import (
    build_score_frame,
    combine_split_scores,
    out_of_sample_scores,
    schema_document,
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)]
    )


def _require_binary_labels(y: pd.Series, split_name: str) -> None:
    classes = sorted(set(np.asarray(y).tolist()))
    if classes != [0, 1]:
        raise ValueError(
            f"{split_name} must contain both classes 0 and 1 for stratified churn "
            f"modeling; observed={classes}."
        )


def _make_models(random_state: int) -> dict:
    return {
        "dummy_prior": DummyClassifier(strategy="prior", random_state=random_state),
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def fit_candidates_on_validation(
    X_train,
    y_train,
    X_valid,
    y_valid,
    *,
    random_state: int = 42,
    threshold_beta: float = 2.0,
) -> dict:
    """Fit candidates on train; score and threshold using validation only."""
    _require_binary_labels(y_train, "train")
    _require_binary_labels(y_valid, "validation")
    fitted = {}
    for name, estimator in _make_models(random_state).items():
        pipe = Pipeline(
            [("preprocess", build_preprocessor(X_train)), ("model", estimator)]
        )
        pipe.fit(X_train, y_train)
        valid_probs = pipe.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold(y_valid, valid_probs, beta=threshold_beta)
        cost_threshold = choose_threshold_cost(y_valid, valid_probs)
        valid_metrics = compute_classification_metrics(y_valid, valid_probs, threshold)
        fitted[name] = {
            "pipeline": pipe,
            "threshold": threshold,
            "cost_threshold": cost_threshold,
            "validation_pr_auc": valid_metrics["pr_auc"],
            "validation_metrics": valid_metrics,
            "model_settings": estimator.get_params(),
        }
    return fitted


def select_model_name(fitted: dict) -> str:
    """Select model by validation PR-AUC only."""
    return max(fitted, key=lambda n: fitted[n]["validation_pr_auc"] or -1.0)


def evaluate_selected_on_test(pipeline, threshold, X_test, y_test) -> dict:
    """Single final evaluation path for the held-out test set."""
    _require_binary_labels(y_test, "test")
    probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, probs, threshold)
    metrics["predictions"] = (probs >= threshold).astype(int)
    metrics["probabilities"] = probs
    return metrics


def _error_analysis(df_test: pd.DataFrame, y_test, probs, preds) -> pd.DataFrame:
    frame = df_test.copy()
    frame["actual"] = np.asarray(y_test)
    frame["predicted"] = preds
    frame["churn_probability"] = probs
    frame["error_type"] = np.where(
        (frame["actual"] == 1) & (frame["predicted"] == 0),
        "false_negative",
        np.where(
            (frame["actual"] == 0) & (frame["predicted"] == 1),
            "false_positive",
            "correct",
        ),
    )
    keep = [
        c
        for c in [
            "customerID",
            "tenure",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "actual",
            "predicted",
            "churn_probability",
            "error_type",
        ]
        if c in frame.columns
    ]
    errors = frame.loc[frame["error_type"] != "correct", keep]
    return errors.sort_values("churn_probability", ascending=False)


def train_and_evaluate(
    df: pd.DataFrame,
    out_dir: str | Path,
    random_state: int = 42,
    *,
    false_positive_cost: float = 25.0,
    false_negative_cost: float = 200.0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    y = (df["Churn"] == "Yes").astype(int)
    _require_binary_labels(y, "full dataset")
    feature_cols = [c for c in df.columns if c not in {"Churn", "customerID"}]
    X = df[feature_cols].copy()

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, random_state=random_state, stratify=y_hold
    )

    fitted = fit_candidates_on_validation(
        X_train, y_train, X_valid, y_valid, random_state=random_state
    )
    # Comparison table: validation metrics only (no test leakage into selection artifacts).
    comparison_rows = []
    for name, info in fitted.items():
        row = {"model": name, "validation_pr_auc": info["validation_pr_auc"]}
        vm = info["validation_metrics"]
        for key in ("precision", "recall", "f1", "roc_auc", "brier_score", "threshold"):
            row[f"validation_{key}"] = vm.get(key)
        row["validation_cost_threshold"] = info["cost_threshold"]
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(out / "model_comparison.csv", index=False)

    selected_name = select_model_name(fitted)
    selected = fitted[selected_name]
    pipe = selected["pipeline"]
    threshold = selected["threshold"]
    cost_threshold = choose_threshold_cost(
        y_valid,
        pipe.predict_proba(X_valid)[:, 1],
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )

    test_eval = evaluate_selected_on_test(pipe, threshold, X_test, y_test)
    probs = test_eval.pop("probabilities")
    preds = test_eval.pop("predictions")
    metrics = dict(test_eval)
    metrics.update(
        {
            "selected_model": selected_name,
            "validation_threshold": threshold,
            "validation_pr_auc_selected": selected["validation_pr_auc"],
            "threshold_selection": "F2 maximization on validation set",
            "cost_threshold_validation": cost_threshold,
            "cost_threshold_assumptions": {
                "false_positive_cost": false_positive_cost,
                "false_negative_cost": false_negative_cost,
                "note": "Illustrative costs; alternative to F2, not used for primary selection.",
            },
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "positive_class": "Churn=Yes",
            "evaluation_protocol": (
                "Model and F2 threshold selected on validation only; "
                "test metrics computed once for the selected model."
            ),
        }
    )
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame(
        classification_report(
            y_test, preds, output_dict=True, zero_division=0, target_names=["No", "Yes"]
        )
    ).T.to_csv(out / "classification_report.csv")

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set(
        title=f"Test confusion matrix — {selected_name}",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label=selected_name)
    ax.set(
        title="Calibration curve (test set)",
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "calibration_curve.png", dpi=150)
    plt.close(fig)

    lift_gains_table(y_test, probs).to_csv(out / "lift_by_decile.csv", index=False)

    # Final descriptive interpretation only; not used for model selection.
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        out / "permutation_importance.csv", index=False
    )

    test_index = X_test.index
    df_test = df.loc[test_index]
    _error_analysis(df_test, y_test, probs, preds).to_csv(
        out / "error_analysis.csv", index=False
    )

    model_version = f"{selected_name}@sklearn-{sklearn_version}"
    train_scores = build_score_frame(
        df.loc[X_train.index],
        pipe.predict_proba(X_train)[:, 1],
        dataset_split="train",
        score_method="in_sample",
        threshold=threshold,
        model_name=selected_name,
        model_version=model_version,
    )
    valid_scores = build_score_frame(
        df.loc[X_valid.index],
        pipe.predict_proba(X_valid)[:, 1],
        dataset_split="validation",
        score_method="out_of_sample",
        threshold=threshold,
        model_name=selected_name,
        model_version=model_version,
    )
    test_scores = build_score_frame(
        df.loc[X_test.index],
        probs,
        dataset_split="test",
        score_method="out_of_sample",
        threshold=threshold,
        model_name=selected_name,
        model_version=model_version,
    )
    all_scores = combine_split_scores([train_scores, valid_scores, test_scores])
    all_scores.to_csv(out / "customer_risk_scores.csv", index=False)
    out_of_sample_scores(all_scores).to_csv(out / "out_of_sample_risk_scores.csv", index=False)
    train_scores.to_csv(out / "train_in_sample_scores.csv", index=False)
    test_scores.to_csv(out / "test_predictions.csv", index=False)
    (out / "scoring_schema.json").write_text(
        json.dumps(schema_document(), indent=2), encoding="utf-8"
    )

    joblib.dump(
        {
            "pipeline": pipe,
            "threshold": threshold,
            "cost_threshold": cost_threshold,
            "model_name": selected_name,
            "model_version": model_version,
        },
        out / "model.joblib",
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "package_versions": {
                    "sklearn": sklearn_version,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
                "random_state": random_state,
                "selected_model": selected_name,
                "model_settings": selected["model_settings"],
                "feature_columns": feature_cols,
                "split": {
                    "train": int(len(X_train)),
                    "validation": int(len(X_valid)),
                    "test": int(len(X_test)),
                },
                "selection": {
                    "model_criterion": "validation PR-AUC",
                    "threshold_criterion": "validation F2",
                    "cost_threshold_alternative": {
                        "false_positive_cost": false_positive_cost,
                        "false_negative_cost": false_negative_cost,
                    },
                },
                "scoring": {
                    "model_version": model_version,
                    "train_score_method": "in_sample",
                    "validation_test_score_method": "out_of_sample",
                    "campaign_default": "out_of_sample_risk_scores.csv",
                },
                "note": (
                    "Threshold and model selection use validation data; "
                    "test set is evaluated once for the selected model. "
                    "Train risk scores are in-sample and labeled as such."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"metrics": metrics, "selected_model": selected_name, "threshold": threshold}
