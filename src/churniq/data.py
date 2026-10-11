from pathlib import Path
import json
import pandas as pd

TARGET = "Churn"
ID_COLUMNS = ["customerID"]

def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/download_dataset.py first.")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} not found.")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df[df[TARGET].isin(["Yes", "No"])].copy()
    if df.empty:
        raise ValueError("No rows with valid Churn labels ('Yes'/'No') were found.")
    return df

def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()) if "customerID" in df else None,
        "missing_values_by_column": {str(k): int(v) for k, v in df.isna().sum().items() if v},
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "churn_rate": round(float((df[TARGET] == "Yes").mean()), 6),
        "columns": list(df.columns),
    }

def save_quality_report(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data_quality_report(df), indent=2), encoding="utf-8")

def make_features_target(df: pd.DataFrame):
    y = (df[TARGET] == "Yes").astype(int)
    drop_cols = [c for c in [TARGET, *ID_COLUMNS] if c in df.columns]
    return df.drop(columns=drop_cols).copy(), y
