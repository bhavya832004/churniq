import numpy as np
import pandas as pd
import pytest

from churniq.modeling import train_and_evaluate
from churniq.scoring import (
    SCORE_SCHEMA,
    build_score_frame,
    combine_split_scores,
    out_of_sample_scores,
    schema_document,
)


def test_score_schema_keys_documented():
    doc = schema_document()
    assert set(SCORE_SCHEMA) == set(doc["schema"])
    frame = build_score_frame(
        pd.DataFrame({"customerID": ["a"], "Churn": ["Yes"]}),
        np.array([0.8]),
        dataset_split="test",
        score_method="out_of_sample",
        threshold=0.5,
        model_name="logistic_regression",
        model_version="logistic_regression@sklearn-1",
    )
    for col in SCORE_SCHEMA:
        assert col in frame.columns


def test_train_cannot_be_labeled_out_of_sample():
    with pytest.raises(ValueError, match="Training rows"):
        build_score_frame(
            pd.DataFrame({"customerID": ["a"], "Churn": ["No"]}),
            np.array([0.2]),
            dataset_split="train",
            score_method="out_of_sample",
            threshold=0.5,
            model_name="m",
            model_version="m@1",
        )


def test_holdout_must_be_out_of_sample():
    with pytest.raises(ValueError, match="out_of_sample"):
        build_score_frame(
            pd.DataFrame({"customerID": ["a"], "Churn": ["No"]}),
            np.array([0.2]),
            dataset_split="validation",
            score_method="in_sample",
            threshold=0.5,
            model_name="m",
            model_version="m@1",
        )


def test_combine_and_filter_out_of_sample():
    train = build_score_frame(
        pd.DataFrame({"customerID": ["t"], "Churn": ["No"]}),
        np.array([0.9]),
        dataset_split="train",
        score_method="in_sample",
        threshold=0.4,
        model_name="m",
        model_version="m@1",
    )
    test = build_score_frame(
        pd.DataFrame({"customerID": ["x"], "Churn": ["Yes"]}),
        np.array([0.7]),
        dataset_split="test",
        score_method="out_of_sample",
        threshold=0.4,
        model_name="m",
        model_version="m@1",
    )
    all_scores = combine_split_scores([train, test])
    oos = out_of_sample_scores(all_scores)
    assert len(all_scores) == 2
    assert list(oos["customerID"]) == ["x"]
    assert (oos["score_method"] == "out_of_sample").all()
    assert int(oos.iloc[0]["threshold_decision"]) == 1


def test_train_and_evaluate_labels_splits(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for i in range(120):
        churn = "Yes" if i % 3 == 0 else "No"
        rows.append(
            {
                "customerID": f"c{i}",
                "tenure": int(rng.integers(1, 72)),
                "MonthlyCharges": float(rng.normal(70, 15)),
                "TotalCharges": float(rng.normal(2000, 500)),
                "Contract": "Month-to-month" if churn == "Yes" else "Two year",
                "InternetService": "DSL",
                "PaymentMethod": "Electronic check",
                "Churn": churn,
            }
        )
    train_and_evaluate(pd.DataFrame(rows), tmp_path, random_state=4)
    scores = pd.read_csv(tmp_path / "customer_risk_scores.csv")
    assert set(scores["dataset_split"]) == {"train", "validation", "test"}
    train_rows = scores.loc[scores["dataset_split"] == "train"]
    holdout = scores.loc[scores["dataset_split"] != "train"]
    assert (train_rows["score_method"] == "in_sample").all()
    assert (holdout["score_method"] == "out_of_sample").all()
    oos = pd.read_csv(tmp_path / "out_of_sample_risk_scores.csv")
    assert (oos["score_method"] == "out_of_sample").all()
    assert "train" not in set(oos["dataset_split"])
    test_pred = pd.read_csv(tmp_path / "test_predictions.csv")
    assert set(test_pred["dataset_split"]) == {"test"}
    assert (tmp_path / "scoring_schema.json").exists()
    assert (tmp_path / "train_in_sample_scores.csv").exists()
