from pathlib import Path

import pandas as pd

from churniq.eda import run_eda, segment_churn_table, wilson_ci


def test_wilson_ci_bounds():
    low, high = wilson_ci(2, 10)
    assert low is not None and high is not None
    assert 0 <= low < 0.2 < high <= 1
    assert wilson_ci(0, 0) == (None, None)


def test_segment_table_includes_denominator_and_ci():
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "Month-to-month", "One year"],
            "Churn": ["Yes", "No", "No"],
        }
    )
    table = segment_churn_table(df, "Contract")
    m2m = table.loc[table["Contract"] == "Month-to-month"].iloc[0]
    assert int(m2m["customers"]) == 2
    assert int(m2m["churners"]) == 1
    assert m2m["churn_rate"] == 0.5
    assert m2m["ci_low_95"] < m2m["churn_rate"] < m2m["ci_high_95"]


def test_run_eda_writes_charts_and_tables(tmp_path: Path):
    df = pd.DataFrame(
        {
            "customerID": list("abcdef"),
            "tenure": [1, 8, 20, 30, 50, 3],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "Two year",
                "Month-to-month",
            ],
            "InternetService": ["Fiber optic", "DSL", "DSL", "No", "DSL", "Fiber optic"],
            "PaymentMethod": ["Electronic check"] * 6,
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "Yes"],
            "MonthlyCharges": [70.0, 55.0, 60.0, 40.0, 45.0, 80.0],
            "TotalCharges": [70.0, 100.0, 600.0, 900.0, 2000.0, 100.0],
            "SeniorCitizen": [0, 0, 1, 0, 1, 0],
            "Churn": ["Yes", "No", "No", "No", "No", "Yes"],
        }
    )
    summary = run_eda(df, tmp_path)
    assert summary["customers"] == 6
    assert summary["churners"] == 2

    for name in [
        "churn_distribution.png",
        "tenure_vs_churn.png",
        "monthly_charges_by_churn.png",
        "churn_rate_by_contract.png",
        "churn_prevalence.csv",
        "churn_by_tenure.csv",
        "churn_by_contract.csv",
        "churn_by_internet.csv",
        "churn_by_payment.csv",
        "churn_by_paperless.csv",
        "numeric_summary.csv",
        "eda_summary.json",
    ]:
        assert (tmp_path / name).exists(), name

    contract = pd.read_csv(tmp_path / "churn_by_contract.csv")
    assert contract["customers"].sum() == 6
    assert {"churners", "churn_rate", "ci_low_95", "ci_high_95"}.issubset(contract.columns)
