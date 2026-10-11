import numpy as np
import pandas as pd
import pytest

from churniq.evaluation import (
    choose_threshold,
    choose_threshold_cost,
    compute_classification_metrics,
)
from churniq.modeling import (
    evaluate_selected_on_test,
    fit_candidates_on_validation,
    select_model_name,
    train_and_evaluate,
)


def test_threshold_range():
    t = choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert 0 <= t <= 1


def test_threshold_selection_uses_only_provided_labels():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    t_valid = choose_threshold(np.array([0, 0, 1, 1]), probs, beta=2.0)
    # Same scores, different labels => different objective surface / threshold.
    t_alt = choose_threshold(np.array([1, 1, 0, 0]), probs, beta=2.0)
    assert 0 <= t_valid <= 1
    assert t_valid != t_alt


def test_metrics_on_toy_fixture():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_classification_metrics(y, p, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert m["pr_auc"] is not None
    assert "accuracy_note" in m


def test_model_selection_uses_validation_pr_auc_only():
    fitted = {
        "a": {"validation_pr_auc": 0.4},
        "b": {"validation_pr_auc": 0.7},
        "c": {"validation_pr_auc": 0.5},
    }
    assert select_model_name(fitted) == "b"


def _binary_frame(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, n),
            "MonthlyCharges": rng.normal(70, 10, n),
            "Contract": rng.choice(["Month-to-month", "One year"], n),
        }
    )
    y = pd.Series([i % 2 for i in range(n)])
    return X, y


def test_fit_candidates_do_not_need_test_set():
    X, y = _binary_frame(80, seed=0)
    X_train, X_valid = X.iloc[:50], X.iloc[50:]
    y_train, y_valid = y.iloc[:50], y.iloc[50:]
    fitted = fit_candidates_on_validation(X_train, y_train, X_valid, y_valid, random_state=0)
    assert set(fitted) >= {"dummy_prior", "logistic_regression", "random_forest"}
    for info in fitted.values():
        assert "validation_pr_auc" in info
        assert "validation_metrics" in info
        assert "metrics" not in info  # no test metrics during fitting/selection


def test_evaluate_selected_on_test_once_shape():
    X, y = _binary_frame(60, seed=1)
    fitted = fit_candidates_on_validation(
        X.iloc[:40], y.iloc[:40], X.iloc[40:50], y.iloc[40:50], random_state=1
    )
    name = select_model_name(fitted)
    metrics = evaluate_selected_on_test(
        fitted[name]["pipeline"],
        fitted[name]["threshold"],
        X.iloc[50:],
        y.iloc[50:],
    )
    assert "precision" in metrics and "recall" in metrics
    assert metrics["confusion_matrix"]


def test_train_and_evaluate_writes_protocol_artifacts(tmp_path):
    rng = np.random.default_rng(2)
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
                "InternetService": rng.choice(["DSL", "Fiber optic"]),
                "PaymentMethod": rng.choice(["Electronic check", "Mailed check"]),
                "Churn": churn,
            }
        )
    df = pd.DataFrame(rows)
    result = train_and_evaluate(df, tmp_path, random_state=2)
    assert result["selected_model"] in {
        "dummy_prior",
        "logistic_regression",
        "random_forest",
    }
    comparison = pd.read_csv(tmp_path / "model_comparison.csv")
    assert "validation_pr_auc" in comparison.columns
    assert not any(c.startswith("test_") for c in comparison.columns)
    assert (tmp_path / "test_metrics.json").exists()
    assert (tmp_path / "error_analysis.csv").exists()
    assert (tmp_path / "lift_by_decile.csv").exists()
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "out_of_sample_risk_scores.csv").exists()
    assert (tmp_path / "test_predictions.csv").exists()


def test_one_class_dataset_fails_clearly(tmp_path):
    df = pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d"],
            "tenure": [1, 2, 3, 4],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
            "Churn": ["No", "No", "No", "No"],
        }
    )
    with pytest.raises(ValueError, match="both classes"):
        train_and_evaluate(df, tmp_path)


def test_cost_threshold_prefers_fewer_false_negatives_when_fn_expensive():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.45, 0.55, 0.6])
    t_fn = choose_threshold_cost(y, p, false_positive_cost=1.0, false_negative_cost=1000.0)
    t_fp = choose_threshold_cost(y, p, false_positive_cost=1000.0, false_negative_cost=1.0)
    assert t_fn <= t_fp
