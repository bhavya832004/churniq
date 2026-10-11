import numpy as np
import pytest

from churniq.business import (
    ASSUMPTION_NOTE,
    campaign_scenarios,
    score_targeting_scenario,
    sensitivity_analysis,
    write_scenario_artifacts,
)


def test_campaign_scenarios_shape_and_labels():
    result = campaign_scenarios(
        [0.1, 0.5, 0.9], customer_value=100, offer_cost=5, offer_success_rate=0.2
    )
    assert len(result) == 6
    assert "scenario_net_value" in result.columns
    assert result.iloc[0]["customers_targeted"] >= 1
    assert bool(result.iloc[0]["not_causal_uplift"]) is True
    assert "assumption" in result.iloc[0]["assumption_note"].lower()


def test_scenario_math_correctness():
    # Target top 1 of 2: p=0.8 -> expected churners 0.8; success 0.5 -> retained 0.4
    # value=100, cost=10 => net = 40 - 10 = 30
    out = score_targeting_scenario(
        [0.2, 0.8],
        count=1,
        customer_value=100,
        offer_cost=10,
        offer_success_rate=0.5,
    )
    assert out["customers_targeted"] == 1
    assert out["expected_churners_without_offer"] == pytest.approx(0.8)
    assert out["assumed_customers_retained"] == pytest.approx(0.4)
    assert out["campaign_cost"] == pytest.approx(10)
    assert out["expected_retained_value"] == pytest.approx(40)
    assert out["scenario_net_value"] == pytest.approx(30)
    assert ASSUMPTION_NOTE in out["assumption_note"]


def test_invalid_success_rate():
    with pytest.raises(ValueError, match="offer_success_rate"):
        campaign_scenarios([0.2], offer_success_rate=2)


def test_negative_costs_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        campaign_scenarios([0.2], customer_value=-1)
    with pytest.raises(ValueError, match="non-negative"):
        campaign_scenarios([0.2], offer_cost=-1)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        campaign_scenarios([1.2, 0.1])
    with pytest.raises(ValueError, match="finite"):
        campaign_scenarios([np.nan])


def test_budget_limits_targeting():
    # offer_cost=25, budget=40 => can afford 1 customer
    result = campaign_scenarios(
        [0.9, 0.8, 0.7], offer_cost=25, budget=40, offer_success_rate=0.2
    )
    assert (result["customers_targeted"] <= 1).all()
    assert int(result["targeting_capacity"].iloc[0]) == 1


def test_sensitivity_varies_assumptions():
    table = sensitivity_analysis([0.1, 0.5, 0.9], base_fraction=0.5)
    assert set(table["varied_parameter"]) >= {
        "assumed_offer_success_rate",
        "assumed_customer_value",
        "assumed_offer_cost",
    }
    assert table["not_causal_uplift"].all()
    success_rows = table.loc[table["varied_parameter"] == "assumed_offer_success_rate"]
    assert success_rows["scenario_net_value"].nunique() > 1


def test_optional_decision_costs():
    probs = np.array([0.9, 0.1])
    labels = np.array([0, 1])  # top target is FP if decision=1
    decisions = np.array([1, 0])
    result = campaign_scenarios(
        probs,
        target_fractions=[0.5],
        false_positive_cost=10,
        false_negative_cost=100,
        labels=labels,
        decisions=decisions,
    )
    assert "illustrative_decision_cost" in result.columns
    assert result.iloc[0]["illustrative_false_positives"] == 1


def test_write_scenario_artifacts(tmp_path):
    paths = write_scenario_artifacts([0.2, 0.8, 0.6], tmp_path)
    assert paths["campaign_scenarios"].exists()
    assert paths["campaign_sensitivity"].exists()
    assert paths["campaign_assumptions"].exists()
    text = paths["campaign_assumptions"].read_text(encoding="utf-8")
    assert "not_causal_uplift" in text
