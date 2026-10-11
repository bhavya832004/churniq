"""Customer risk scoring with explicit split / score-method labels."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SCORE_SCHEMA: dict[str, str] = {
    "customerID": "Join key only; never used as a model feature.",
    "Churn": "Observed label when available (teaching dataset).",
    "dataset_split": "train | validation | test",
    "score_method": (
        "in_sample (fit on same training rows; not unbiased) | "
        "out_of_sample (validation/test holdout after training)"
    ),
    "churn_probability": "Model estimated P(Churn=Yes).",
    "risk_tier": "Low | Moderate | High | Very high from probability bins.",
    "threshold_decision": "1 if probability >= operating threshold else 0.",
    "operating_threshold": "Validation-selected F2 threshold used for decisions.",
    "model_name": "Selected model name.",
    "model_version": "Package/model version string for this run.",
    "scored_at_utc": "UTC timestamp when scores were generated.",
    "disclaimer": "Not real customer advice; public teaching data only.",
}

DISCLAIMER = (
    "Not real customer advice. Public teaching dataset scores are for portfolio "
    "demonstration only; train in-sample scores are not unbiased predictions."
)

RISK_BINS = [-0.001, 0.25, 0.5, 0.75, 1.001]
RISK_LABELS = ["Low", "Moderate", "High", "Very high"]


def assign_risk_tier(probabilities: pd.Series | np.ndarray) -> pd.Series:
    return pd.cut(
        np.asarray(probabilities, dtype=float),
        bins=RISK_BINS,
        labels=RISK_LABELS,
    ).astype(str)


def build_score_frame(
    df_slice: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    dataset_split: str,
    score_method: str,
    threshold: float,
    model_name: str,
    model_version: str,
    scored_at_utc: str | None = None,
) -> pd.DataFrame:
    if dataset_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown dataset_split: {dataset_split}")
    if score_method not in {"in_sample", "out_of_sample"}:
        raise ValueError(f"Unknown score_method: {score_method}")
    if dataset_split == "train" and score_method == "out_of_sample":
        raise ValueError("Training rows must not be labeled out_of_sample without OOF scores.")
    if dataset_split in {"validation", "test"} and score_method != "out_of_sample":
        raise ValueError("Validation/test holdout scores must be labeled out_of_sample.")

    scored_at = scored_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    probs = np.asarray(probabilities, dtype=float)
    if len(df_slice) != len(probs):
        raise ValueError("df_slice and probabilities length mismatch")

    out = pd.DataFrame(index=df_slice.index)
    if "customerID" in df_slice.columns:
        out["customerID"] = df_slice["customerID"].to_numpy()
    if "Churn" in df_slice.columns:
        out["Churn"] = df_slice["Churn"].to_numpy()
    out["dataset_split"] = dataset_split
    out["score_method"] = score_method
    out["churn_probability"] = probs
    out["risk_tier"] = assign_risk_tier(probs)
    out["operating_threshold"] = float(threshold)
    out["threshold_decision"] = (probs >= threshold).astype(int)
    out["model_name"] = model_name
    out["model_version"] = model_version
    out["scored_at_utc"] = scored_at
    out["disclaimer"] = DISCLAIMER
    return out


def combine_split_scores(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(SCORE_SCHEMA))
    return pd.concat(frames, axis=0).sort_index()


def out_of_sample_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rows safe to treat as non-training predictions for demos/campaigns."""
    return scores.loc[scores["score_method"] == "out_of_sample"].copy()


def schema_document() -> dict[str, Any]:
    return {
        "schema": SCORE_SCHEMA,
        "primary_artifact": "customer_risk_scores.csv",
        "unbiased_artifact": "out_of_sample_risk_scores.csv",
        "test_artifact": "test_predictions.csv",
        "train_artifact": "train_in_sample_scores.csv",
        "notes": [
            "customerID is export/join only and excluded from model features.",
            "Train scores are in-sample and must not be treated as unbiased.",
            "Validation/test scores are out-of-sample relative to the fitted training pipeline.",
        ],
    }
