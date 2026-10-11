from pathlib import Path
import json

import pandas as pd

from .validation import (
    ID_COLUMNS,
    TARGET,
    DataValidationError,
    prepare_dataframe,
    build_data_quality_report,
    validate_prepared_frame,
)

__all__ = [
    "TARGET",
    "ID_COLUMNS",
    "DataValidationError",
    "load_dataset",
    "load_dataset_with_report",
    "data_quality_report",
    "save_quality_report",
    "make_features_target",
]


def load_dataset_with_report(
    path: str | Path,
    *,
    raise_on_error: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load a CSV, prepare types, and return (frame, quality report).

    Does not drop rows for invalid labels or conversion failures. Fatal
    problems are recorded in ``report['errors']`` and raised when
    ``raise_on_error`` is True.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run scripts/download_dataset.py first."
        )

    file_size_bytes = int(path.stat().st_size)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas parser / empty-file failures
        report = {
            "ok": False,
            "source_path": str(path),
            "errors": [f"Failed to parse CSV at {path}: {exc}"],
            "warnings": [],
            "metrics": {
                "row_count": 0,
                "column_count": 0,
                "file_size_bytes": file_size_bytes,
            },
            "preprocessing_notes": [],
            "row_count": 0,
            "column_count": 0,
            "duplicate_rows": 0,
            "duplicate_customer_ids": None,
            "missing_values_by_column": {},
            "target_counts": {},
            "churn_rate": None,
            "columns": [],
        }
        if raise_on_error:
            raise DataValidationError(report["errors"][0], report) from exc
        return pd.DataFrame(), report

    prepared, notes, coerced_missing = prepare_dataframe(df)
    report = validate_prepared_frame(
        prepared,
        source_path=path,
        preprocessing_notes=notes,
        total_charges_coerced_to_missing=coerced_missing,
        file_size_bytes=file_size_bytes,
        raise_on_error=raise_on_error,
    )
    return prepared, report


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a dataset, raising on fatal data-contract errors."""
    df, _report = load_dataset_with_report(path, raise_on_error=True)
    return df


def data_quality_report(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    preprocessing_notes: list[str] | None = None,
    total_charges_coerced_to_missing: int = 0,
    file_size_bytes: int | None = None,
) -> dict:
    return build_data_quality_report(
        df,
        source_path=source_path,
        preprocessing_notes=preprocessing_notes,
        total_charges_coerced_to_missing=total_charges_coerced_to_missing,
        file_size_bytes=file_size_bytes,
    )


def save_quality_report(report_or_df, path: str | Path) -> None:
    """Save a quality report dict, or build one from a DataFrame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(report_or_df, pd.DataFrame):
        payload = data_quality_report(report_or_df)
    else:
        payload = report_or_df
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_features_target(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} not found.")
    invalid = ~df[TARGET].isin(["Yes", "No"])
    if invalid.any():
        raise DataValidationError(
            "Cannot build features/target while invalid Churn labels remain.",
            data_quality_report(df),
        )
    y = (df[TARGET] == "Yes").astype(int)
    drop_cols = [c for c in [TARGET, *ID_COLUMNS] if c in df.columns]
    return df.drop(columns=drop_cols).copy(), y
