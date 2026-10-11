"""Helpers for loading dashboard artifacts (used by Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def mask_customer_id(series: pd.Series) -> pd.Series:
    def _mask(value: object) -> str:
        text = str(value)
        if len(text) <= 4:
            return "***"
        return text[:2] + "***" + text[-2:]

    return series.astype(str).map(_mask)


def select_out_of_sample_scores(all_scores: pd.DataFrame) -> pd.DataFrame:
    if "score_method" in all_scores.columns:
        return all_scores.loc[all_scores["score_method"] == "out_of_sample"].copy()
    if "dataset_split" in all_scores.columns:
        return all_scores.loc[all_scores["dataset_split"] != "train"].copy()
    return all_scores.copy()
