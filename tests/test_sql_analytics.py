from pathlib import Path

import pandas as pd

from churniq.sql_analytics import SQL_QUERIES, run_sql_analytics


def _tiny_df():
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d", "e", "f"],
            "tenure": [1, 8, 20, 30, 50, 3],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "Two year",
                "Month-to-month",
            ],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
                "Electronic check",
            ],
            "MonthlyCharges": [70.0, 55.0, 60.0, 40.0, 45.0, 80.0],
            "TotalCharges": [70.0, None, 600.0, 900.0, 2000.0, 100.0],
            "Churn": ["Yes", "No", "No", "No", "No", "Yes"],
        }
    )


def test_sql_analytics_writes_expected_artifacts(tmp_path: Path):
    db = run_sql_analytics(_tiny_df(), tmp_path)
    assert db.exists()
    assert (tmp_path / "sql_queries.sql").exists()
    for name in SQL_QUERIES:
        path = tmp_path / f"{name}.csv"
        assert path.exists(), name
        frame = pd.read_csv(path)
        assert len(frame) >= 1

    contract = pd.read_csv(tmp_path / "sql_contract_analysis.csv")
    assert {"customers", "churners", "churn_rate"}.issubset(contract.columns)
    assert contract["customers"].sum() == 6

    ranked = pd.read_csv(tmp_path / "sql_contract_churn_ranked.csv")
    assert "churn_rate_rank" in ranked.columns

    validation = pd.read_csv(tmp_path / "sql_validation_checks.csv").iloc[0]
    assert int(validation["row_count"]) == 6
    assert int(validation["null_total_charges"]) == 1


def test_sql_outputs_derived_from_input_not_constants(tmp_path: Path):
    df = _tiny_df()
    run_sql_analytics(df, tmp_path)
    cohorts = pd.read_csv(tmp_path / "sql_tenure_cohorts.csv")
    assert cohorts["customers"].sum() == len(df)
