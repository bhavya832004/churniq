import json
import subprocess
import sys

import pandas as pd
import pytest
from churniq.data import load_dataset


def test_cli_audit_writes_json(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    out_path = tmp_path / "quality.json"
    pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [1, 2],
            "TotalCharges": ["10", "20"],
            "Churn": ["Yes", "No"],
        }
    ).to_csv(csv_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "churniq.cli",
            "audit",
            "--data",
            str(csv_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["metrics"]["row_count"] == 2
    assert report["metrics"]["churn_rate"] == 0.5
    assert report["source_path"]
    assert isinstance(report["preprocessing_notes"], list)


def test_cli_audit_fails_on_invalid_target(tmp_path):
    csv_path = tmp_path / "bad.csv"
    out_path = tmp_path / "quality.json"
    pd.DataFrame({"customerID": ["a"], "Churn": ["Maybe"]}).to_csv(csv_path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "churniq.cli",
            "audit",
            "--data",
            str(csv_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["errors"]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.csv")
