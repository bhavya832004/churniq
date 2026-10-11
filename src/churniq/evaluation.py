"""Metric helpers and threshold selection (validation-only objectives)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, probabilities, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    classes = set(np.unique(y_true).tolist())

    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "accuracy_note": (
            "Accuracy alone is inadequate under class imbalance: a model that always "
            "predicts the majority class can look strong while missing churners."
        ),
    }
    if classes == {0, 1}:
        result["pr_auc"] = float(average_precision_score(y_true, probabilities))
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
        result["metric_warning"] = (
            "ROC-AUC/PR-AUC undefined or skipped because the label set is not binary "
            f"with both classes present (observed={sorted(classes)})."
        )
    return result


def choose_threshold(y_true, probabilities, beta: float = 2.0) -> float:
    """Select threshold by maximizing F-beta on the provided labels only."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    b2 = beta**2
    denom = b2 * precision[:-1] + recall[:-1]
    fbeta = np.divide(
        (1 + b2) * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    return float(thresholds[int(np.argmax(fbeta))])


def choose_threshold_cost(
    y_true,
    probabilities,
    *,
    false_positive_cost: float = 25.0,
    false_negative_cost: float = 200.0,
) -> float:
    """Select threshold minimizing expected FP/FN cost on provided labels only."""
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Costs must be non-negative")
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.clip(probabilities, 0, 1))
    if len(candidates) == 0:
        return 0.5
    best_t, best_cost = 0.5, float("inf")
    for t in candidates:
        pred = probabilities >= t
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp * false_positive_cost + fn * false_negative_cost
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def lift_gains_table(y_true, probabilities, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"actual": np.asarray(y_true), "probability": np.asarray(probabilities, dtype=float)}
    ).sort_values("probability", ascending=False, kind="mergesort")
    n = len(frame)
    if n == 0:
        return pd.DataFrame()
    bins = min(n_bins, n)
    frame["risk_decile"] = pd.qcut(
        frame["probability"].rank(method="first"), q=bins, labels=False, duplicates="drop"
    ) + 1
    table = (
        frame.groupby("risk_decile", observed=False)
        .agg(
            customers=("actual", "size"),
            mean_predicted_risk=("probability", "mean"),
            observed_churn_rate=("actual", "mean"),
            churners=("actual", "sum"),
        )
        .reset_index()
        .sort_values("risk_decile", ascending=False)
    )
    overall = float(frame["actual"].mean()) if n else 0.0
    table["lift_vs_average"] = table["observed_churn_rate"] / overall if overall else np.nan
    table["cumulative_customers"] = table["customers"].cumsum()
    table["cumulative_churners"] = table["churners"].cumsum()
    total_churners = float(frame["actual"].sum())
    table["cumulative_gain"] = (
        table["cumulative_churners"] / total_churners if total_churners else np.nan
    )
    table["cumulative_customers_share"] = table["cumulative_customers"] / n
    return table
