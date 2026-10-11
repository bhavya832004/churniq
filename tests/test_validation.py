import json

import pandas as pd
import pytest

from churniq.data import load_dataset, load_dataset_with_report, data_quality_report
from churniq.validation import DataValidationError, build_data_quality_report


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_blank_total_charges_coerced_and_retained(tmp_path):
    path = tmp_path / "blank_charges.csv"
    _write_csv(
        path,
        {
            "customerID": ["a", "b", "c"],
            "tenure": [1, 2, 3],
            "TotalCharges": ["10", " ", "30"],
            "Churn": ["Yes", "No", "No"],
        },
    )
    df, report = load_dataset_with_report(path)
    assert len(df) == 3
    assert df["TotalCharges"].isna().sum() == 1
    assert report["metrics"]["total_charges_coerced_to_missing"] == 1
    assert report["ok"] is True
    assert any("TotalCharges" in note for note in report["preprocessing_notes"])
    assert any("retained" in note.lower() for note in report["preprocessing_notes"])


def test_missing_target_column_is_error(tmp_path):
    path = tmp_path / "no_target.csv"
    _write_csv(path, {"customerID": ["a"], "tenure": [1]})
    with pytest.raises(DataValidationError) as excinfo:
        load_dataset(path)
    assert "Churn" in str(excinfo.value)
    assert excinfo.value.report["ok"] is False
    assert excinfo.value.report["errors"]


def test_invalid_target_values_fail_without_dropping(tmp_path):
    path = tmp_path / "bad_labels.csv"
    _write_csv(
        path,
        {
            "customerID": ["a", "b", "c"],
            "Churn": ["Yes", "Maybe", "No"],
        },
    )
    with pytest.raises(DataValidationError) as excinfo:
        load_dataset(path)
    report = excinfo.value.report
    assert report["metrics"]["row_count"] == 3
    assert "Maybe" in report["metrics"]["invalid_target_counts"]
    assert any("were not dropped" in err.lower() for err in report["errors"])


def test_empty_dataset_is_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("customerID,Churn\n", encoding="utf-8")
    with pytest.raises(DataValidationError) as excinfo:
        load_dataset(path)
    assert any("zero rows" in err.lower() for err in excinfo.value.report["errors"])


def test_malformed_csv_is_error(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError) as excinfo:
        load_dataset(path)
    assert excinfo.value.report["ok"] is False
    assert excinfo.value.report["errors"]


def test_duplicate_ids_are_warning_not_silent_drop(tmp_path):
    path = tmp_path / "dup_ids.csv"
    _write_csv(
        path,
        {
            "customerID": ["a", "a", "b"],
            "Churn": ["Yes", "No", "No"],
        },
    )
    df, report = load_dataset_with_report(path)
    assert len(df) == 3
    assert report["metrics"]["duplicate_customer_ids"] == 1
    assert any("duplicate customerID" in w for w in report["warnings"])


def test_one_class_target_warns(tmp_path):
    path = tmp_path / "one_class.csv"
    _write_csv(
        path,
        {
            "customerID": ["a", "b", "c"],
            "Churn": ["No", "No", "No"],
        },
    )
    df, report = load_dataset_with_report(path)
    assert len(df) == 3
    assert report["ok"] is True
    assert report["metrics"]["churn_rate"] == 0.0
    assert any("One-class target" in w for w in report["warnings"])


def test_report_has_structured_contract_fields():
    report = build_data_quality_report(
        pd.DataFrame({"customerID": ["a", "b"], "Churn": ["Yes", "No"]}),
        source_path="memory://fixture",
        preprocessing_notes=["fixture note"],
    )
    assert "errors" in report and "warnings" in report and "metrics" in report
    assert report["source_path"] == "memory://fixture"
    assert report["preprocessing_notes"] == ["fixture note"]
    assert "schema" in report["metrics"]
    assert report["metrics"]["row_count"] == 2
    # JSON-serializable
    json.dumps(report)


def test_data_quality_report_wrapper_matches():
    df = pd.DataFrame({"customerID": ["a", "a"], "Churn": ["Yes", "No"]})
    report = data_quality_report(df)
    assert report["row_count"] == 2
    assert report["duplicate_customer_ids"] == 1
    assert report["metrics"]["duplicate_customer_ids"] == 1
