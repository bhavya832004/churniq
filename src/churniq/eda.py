"""Exploratory analysis and segment summaries (non-interactive)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None, None
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def segment_churn_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate by segment with denominator and 95% Wilson CI."""
    if column not in df.columns:
        return pd.DataFrame(
            columns=[
                column,
                "customers",
                "churners",
                "churn_rate",
                "ci_low_95",
                "ci_high_95",
            ]
        )
    rows = []
    for key, group in df.groupby(column, dropna=False, observed=False):
        n = int(len(group))
        churners = int((group["Churn"] == "Yes").sum())
        low, high = wilson_ci(churners, n)
        rows.append(
            {
                column: key if pd.notna(key) else "<NA>",
                "customers": n,
                "churners": churners,
                "churn_rate": round(churners / n, 6) if n else None,
                "ci_low_95": None if low is None else round(low, 6),
                "ci_high_95": None if high is None else round(high, 6),
            }
        )
    return pd.DataFrame(rows).sort_values("churn_rate", ascending=False, na_position="last")


def _save_bar(counts: pd.Series, *, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#3B6D9A")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    for i, val in enumerate(counts.values):
        ax.text(i, val, f"{int(val)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """Create EDA charts/tables from the loaded frame. Returns a small summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    n = len(df)
    churners = int((df["Churn"] == "Yes").sum())
    churn_rate = float(churners / n) if n else float("nan")
    summary = {
        "customers": n,
        "churners": churners,
        "churn_rate": round(churn_rate, 6) if n else None,
        "class_imbalance_ratio_no_to_yes": round((n - churners) / churners, 4)
        if churners
        else None,
        "note": "Segment associations are observational; not causal effects.",
    }

    counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    _save_bar(
        counts,
        title=f"Churn prevalence (n={n}, rate={churn_rate:.1%})",
        xlabel="Churn label",
        ylabel="Customers (count)",
        path=out / "churn_distribution.png",
    )
    pd.DataFrame(
        {
            "Churn": ["No", "Yes"],
            "customers": [int(counts["No"]), int(counts["Yes"])],
            "share": [
                round(counts["No"] / n, 6) if n else None,
                round(counts["Yes"] / n, 6) if n else None,
            ],
        }
    ).to_csv(out / "churn_prevalence.csv", index=False)

    if "tenure" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(
            data=df,
            x="tenure",
            hue="Churn",
            bins=24,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set(
            title="Tenure distribution by churn (density, observational)",
            xlabel="Tenure (months)",
            ylabel="Density",
        )
        fig.tight_layout()
        fig.savefig(out / "tenure_vs_churn.png", dpi=150)
        plt.close(fig)

        tenure = df.assign(
            tenure_band=pd.cut(
                df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
            )
        )
        segment_churn_table(tenure, "tenure_band").to_csv(
            out / "churn_by_tenure.csv", index=False
        )

    for column, filename in [
        ("Contract", "churn_by_contract.csv"),
        ("InternetService", "churn_by_internet.csv"),
        ("PaymentMethod", "churn_by_payment.csv"),
        ("PaperlessBilling", "churn_by_paperless.csv"),
    ]:
        if column in df.columns:
            segment_churn_table(df, column).to_csv(out / filename, index=False)

    numeric_cols = [
        c
        for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        if c in df.columns
    ]
    if numeric_cols:
        desc = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        desc = desc.rename_axis("feature").reset_index()
        # Simple outlier fence counts (IQR rule) for context, not removal.
        outlier_rows = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                outlier_rows.append({"feature": col, "iqr_outlier_count": 0})
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or not np.isfinite(iqr):
                outlier_rows.append({"feature": col, "iqr_outlier_count": 0})
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_rows.append(
                {
                    "feature": col,
                    "iqr_outlier_count": int(((series < low) | (series > high)).sum()),
                }
            )
        desc.merge(pd.DataFrame(outlier_rows), on="feature", how="left").to_csv(
            out / "numeric_summary.csv", index=False
        )

    if "MonthlyCharges" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = [
            df.loc[df["Churn"] == label, "MonthlyCharges"].dropna().to_numpy()
            for label in ("No", "Yes")
        ]
        ax.boxplot(data, tick_labels=["No", "Yes"], patch_artist=True)
        ax.set(
            title="Monthly charges by churn (USD / month)",
            xlabel="Churn label",
            ylabel="MonthlyCharges (USD)",
        )
        fig.tight_layout()
        fig.savefig(out / "monthly_charges_by_churn.png", dpi=150)
        plt.close(fig)

    # Segment rate chart for contract when available (rates + n labels).
    if "Contract" in df.columns:
        contract = segment_churn_table(df, "Contract")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.barplot(data=contract, x="Contract", y="churn_rate", ax=ax, color="#3B6D9A")
        ax.set(
            title="Observed churn rate by contract (with customer counts)",
            xlabel="Contract",
            ylabel="Churn rate",
        )
        ax.set_ylim(0, min(1.0, float(contract["churn_rate"].max() or 0) + 0.15))
        for i, row in contract.reset_index(drop=True).iterrows():
            ax.text(
                i,
                row["churn_rate"],
                f"n={int(row['customers'])}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(out / "churn_rate_by_contract.png", dpi=150)
        plt.close(fig)

    pd.Series(summary).to_json(out / "eda_summary.json")
    return summary
