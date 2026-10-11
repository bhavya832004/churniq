from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme()
    counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set(title="Customer churn distribution", xlabel="Churn", ylabel="Customers")
    fig.tight_layout(); fig.savefig(out / "churn_distribution.png", dpi=150); plt.close(fig)
    if "tenure" in df:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="tenure", hue="Churn", bins=24, stat="density",
                     common_norm=False, element="step", ax=ax)
        ax.set(title="Tenure distribution by churn", xlabel="Tenure (months)")
        fig.tight_layout(); fig.savefig(out / "tenure_vs_churn.png", dpi=150); plt.close(fig)
        tenure = df.assign(tenure_band=pd.cut(df["tenure"], bins=[-1,6,12,24,48,72],
            labels=["0-6","7-12","13-24","25-48","49-72"]))
        tenure.groupby("tenure_band", observed=False)["Churn"].apply(
            lambda s: (s == "Yes").mean()).rename("churn_rate").reset_index().to_csv(
            out / "churn_by_tenure.csv", index=False)
    if "Contract" in df:
        pd.crosstab(df["Contract"], df["Churn"], normalize="index").reset_index().to_csv(
            out / "churn_by_contract.csv", index=False)
