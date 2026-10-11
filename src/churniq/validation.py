"""Structured dataset validation and data-quality reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Churn"
VALID_TARGET_VALUES = frozenset({"Yes", "No"})
ID_COLUMNS = ("customerID",)

# Columns commonly present in the IBM Telco teaching dataset.
OPTIONAL_COLUMNS = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)


class DataValidationError(ValueError):
    """Raised when dataset validation finds fatal problems."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _schema_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
            }
        )
    return rows


def _non_finite_numeric_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in df.select_dtypes(include=["number"]).columns:
        mask = df[col].notna()
        if not mask.any():
            continue
        non_finite = int((~np.isfinite(df.loc[mask, col].astype(float))).sum())
        if non_finite:
            counts[str(col)] = non_finite
    return counts


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Normalize types without dropping rows.

    Returns the prepared frame, preprocessing notes, and TotalCharges
    values coerced to missing.
    """
    out = df.copy()
    notes: list[str] = []
    total_charges_to_missing = 0

    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype("string").str.strip()
        notes.append("Stripped whitespace from Churn labels and cast to string.")

    if "TotalCharges" in out.columns:
        before_missing = int(out["TotalCharges"].isna().sum())
        raw = out["TotalCharges"]
        # Treat blank/whitespace strings as conversion failures before coercion.
        if raw.dtype == object or str(raw.dtype) == "string":
            blank_mask = raw.astype("string").str.strip().isin(["", "<NA>", "nan", "None"])
            blank_count = int(blank_mask.fillna(False).sum())
        else:
            blank_count = 0
        coerced = pd.to_numeric(out["TotalCharges"], errors="coerce")
        after_missing = int(coerced.isna().sum())
        total_charges_to_missing = max(0, after_missing - before_missing)
        out["TotalCharges"] = coerced
        notes.append(
            "Coerced TotalCharges to numeric with errors='coerce'; "
            f"{total_charges_to_missing} value(s) became missing "
            f"({blank_count} blank/whitespace-like)."
        )
        notes.append("Rows with missing TotalCharges were retained.")

    notes.append("No rows were dropped during preparation.")
    return out, notes, total_charges_to_missing


def build_data_quality_report(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    preprocessing_notes: list[str] | None = None,
    total_charges_coerced_to_missing: int = 0,
    file_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Return structured errors, warnings, and metrics for a dataset."""
    errors: list[str] = []
    warnings: list[str] = []
    notes = list(preprocessing_notes or [])

    source_display = None if source_path is None else str(source_path)
    path = Path(source_path) if source_path is not None else None
    if file_size_bytes is None and path is not None:
        try:
            if path.exists():
                file_size_bytes = int(path.stat().st_size)
        except OSError:
            # Non-filesystem source strings (e.g. fixtures) should not break reporting.
            pass


    row_count = int(len(df))
    column_count = int(df.shape[1])

    if row_count == 0:
        errors.append("Dataset has zero rows.")

    if TARGET not in df.columns:
        errors.append(f"Required target column {TARGET!r} is missing.")
        target_counts: dict[str, int] = {}
        churn_rate = None
        invalid_target_counts: dict[str, int] = {}
    else:
        labels = df[TARGET].astype("string")
        value_counts = labels.value_counts(dropna=False)
        target_counts = {str(k): int(v) for k, v in value_counts.items()}
        invalid_mask = ~labels.isin(list(VALID_TARGET_VALUES))
        # Missing labels are invalid for supervised churn modeling.
        invalid_target_counts = {
            str(k): int(v)
            for k, v in labels[invalid_mask].value_counts(dropna=False).items()
        }
        if invalid_target_counts:
            details = ", ".join(f"{k!r}: {v}" for k, v in invalid_target_counts.items())
            errors.append(
                "Target column Churn contains values other than 'Yes'/'No' "
                f"(including missing). Counts: {details}. "
                "Rows were not dropped; fix or filter upstream explicitly."
            )
        valid_n = int(labels.isin(list(VALID_TARGET_VALUES)).sum())
        if valid_n == 0 and row_count > 0:
            errors.append("No rows with valid Churn labels ('Yes'/'No') were found.")
        yes_n = int((labels == "Yes").sum())
        no_n = int((labels == "No").sum())
        if valid_n > 0 and (yes_n == 0 or no_n == 0):
            present = "Yes" if yes_n else "No"
            warnings.append(
                f"One-class target detected: only {present!r} labels are present "
                f"among valid Churn values (n={valid_n}). Stratified modeling will fail."
            )
        churn_rate = round(float(yes_n / valid_n), 6) if valid_n else None

    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if missing_optional:
        warnings.append(
            "Optional / expected Telco columns missing: " + ", ".join(missing_optional)
        )

    for id_col in ID_COLUMNS:
        if id_col not in df.columns:
            warnings.append(f"Identifier column {id_col!r} is missing.")

    duplicate_rows = int(df.duplicated().sum()) if row_count else 0
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate row(s). Rows were retained.")

    duplicate_ids = None
    if "customerID" in df.columns:
        duplicate_ids = int(df["customerID"].duplicated().sum())
        if duplicate_ids:
            warnings.append(
                f"Found {duplicate_ids} duplicate customerID value(s). Rows were retained."
            )

    missing_by_column = {str(k): int(v) for k, v in df.isna().sum().items() if v}
    if missing_by_column:
        warnings.append(
            "Missing values present in columns: "
            + ", ".join(f"{k} ({v})" for k, v in missing_by_column.items())
        )

    if total_charges_coerced_to_missing:
        warnings.append(
            f"TotalCharges coercion produced {total_charges_coerced_to_missing} "
            "additional missing value(s); rows retained."
        )

    non_finite = _non_finite_numeric_counts(df)
    if non_finite:
        warnings.append(
            "Non-finite numeric values found: "
            + ", ".join(f"{k} ({v})" for k, v in non_finite.items())
        )

    metrics = {
        "row_count": row_count,
        "column_count": column_count,
        "file_size_bytes": file_size_bytes,
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": duplicate_ids,
        "missing_values_by_column": missing_by_column,
        "target_counts": target_counts,
        "invalid_target_counts": invalid_target_counts if TARGET in df.columns else {},
        "churn_rate": churn_rate,
        "total_charges_coerced_to_missing": int(total_charges_coerced_to_missing),
        "non_finite_numeric_by_column": non_finite,
        "missing_optional_columns": missing_optional,
        "columns": [str(c) for c in df.columns],
        "schema": _schema_summary(df),
    }

    report = {
        "ok": not errors,
        "source_path": source_display,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
        "preprocessing_notes": notes,
        # Convenience aliases used by existing callers/tests.
        "row_count": metrics["row_count"],
        "column_count": metrics["column_count"],
        "duplicate_rows": metrics["duplicate_rows"],
        "duplicate_customer_ids": metrics["duplicate_customer_ids"],
        "missing_values_by_column": metrics["missing_values_by_column"],
        "target_counts": metrics["target_counts"],
        "churn_rate": metrics["churn_rate"],
        "columns": metrics["columns"],
    }
    return report


def validate_prepared_frame(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    preprocessing_notes: list[str] | None = None,
    total_charges_coerced_to_missing: int = 0,
    file_size_bytes: int | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    report = build_data_quality_report(
        df,
        source_path=source_path,
        preprocessing_notes=preprocessing_notes,
        total_charges_coerced_to_missing=total_charges_coerced_to_missing,
        file_size_bytes=file_size_bytes,
    )
    if raise_on_error and report["errors"]:
        joined = " ".join(report["errors"])
        raise DataValidationError(joined, report)
    return report
