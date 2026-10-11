"""Illustrative retention campaign scenario calculator (not causal uplift)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASSUMPTION_NOTE = (
    "SCENARIO ASSUMPTIONS ONLY: retained customers are assumed as "
    "sum(churn_probability) * offer_success_rate. This is not a causal uplift "
    "or individual treatment effect estimate. A churn model alone cannot tell "
    "who will change behavior because of an offer."
)

DEFAULTS = {
    "customer_value": 500.0,
    "offer_cost": 25.0,
    "offer_success_rate": 0.20,
    "target_fractions": (0.05, 0.10, 0.20, 0.30, 0.50, 1.0),
}


def _validate_inputs(
    probabilities,
    *,
    customer_value: float,
    offer_cost: float,
    offer_success_rate: float,
    max_customers: int | None,
    budget: float | None,
    false_positive_cost: float | None,
    false_negative_cost: float | None,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1:
        raise ValueError("probabilities must be a 1-d array-like")
    if len(p) and (np.isnan(p).any() or np.isinf(p).any()):
        raise ValueError("probabilities must be finite")
    if len(p) and ((p < 0).any() or (p > 1).any()):
        raise ValueError("probabilities must be within [0, 1]")
    if customer_value < 0 or offer_cost < 0:
        raise ValueError("customer_value and offer_cost must be non-negative")
    if not 0 <= offer_success_rate <= 1:
        raise ValueError("offer_success_rate must be between 0 and 1")
    if max_customers is not None and max_customers < 0:
        raise ValueError("max_customers must be non-negative")
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative")
    if false_positive_cost is not None and false_positive_cost < 0:
        raise ValueError("false_positive_cost must be non-negative")
    if false_negative_cost is not None and false_negative_cost < 0:
        raise ValueError("false_negative_cost must be non-negative")
    return p


def _capacity(
    n: int,
    *,
    max_customers: int | None,
    budget: float | None,
    offer_cost: float,
) -> int:
    cap = n
    if max_customers is not None:
        cap = min(cap, int(max_customers))
    if budget is not None:
        if offer_cost == 0:
            afford = n
        else:
            afford = int(budget // offer_cost)
        cap = min(cap, afford)
    return max(0, cap)


def score_targeting_scenario(
    probabilities,
    count: int,
    *,
    customer_value: float = DEFAULTS["customer_value"],
    offer_cost: float = DEFAULTS["offer_cost"],
    offer_success_rate: float = DEFAULTS["offer_success_rate"],
) -> dict:
    """Score one targeting count under explicit scenario assumptions."""
    p = _validate_inputs(
        probabilities,
        customer_value=customer_value,
        offer_cost=offer_cost,
        offer_success_rate=offer_success_rate,
        max_customers=None,
        budget=None,
        false_positive_cost=None,
        false_negative_cost=None,
    )
    n = len(p)
    count = int(max(0, min(count, n)))
    order = np.argsort(-p)
    selected = order[:count]
    expected_churners = float(p[selected].sum()) if count else 0.0
    assumed_retained = expected_churners * offer_success_rate
    cost = count * offer_cost
    retained_value = assumed_retained * customer_value
    return {
        "customers_targeted": count,
        "mean_churn_probability": float(p[selected].mean()) if count else 0.0,
        "expected_churners_without_offer": expected_churners,
        "assumed_customers_retained": assumed_retained,
        "campaign_cost": cost,
        "expected_retained_value": retained_value,
        "scenario_net_value": retained_value - cost,
        "assumption_note": ASSUMPTION_NOTE,
        "defaults_are_illustrative": True,
    }


def campaign_scenarios(
    probabilities,
    customer_value: float = DEFAULTS["customer_value"],
    offer_cost: float = DEFAULTS["offer_cost"],
    offer_success_rate: float = DEFAULTS["offer_success_rate"],
    max_customers: int | None = None,
    budget: float | None = None,
    target_fractions: tuple[float, ...] | list[float] | None = None,
    false_positive_cost: float | None = None,
    false_negative_cost: float | None = None,
    labels: np.ndarray | None = None,
    decisions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build scenario rows across target fractions under assumed success rates.

    Optional FP/FN costs are illustrative decision-cost add-ons when labels and
    threshold decisions are supplied; they are not causal campaign effects.
    """
    p = _validate_inputs(
        probabilities,
        customer_value=customer_value,
        offer_cost=offer_cost,
        offer_success_rate=offer_success_rate,
        max_customers=max_customers,
        budget=budget,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    fractions = list(target_fractions or DEFAULTS["target_fractions"])
    n = len(p)
    capacity = _capacity(
        n, max_customers=max_customers, budget=budget, offer_cost=offer_cost
    )
    order = np.argsort(-p)
    rows = []
    for fraction in fractions:
        if n == 0 or capacity == 0:
            count = 0
        else:
            count = min(max(1, int(np.ceil(n * float(fraction)))), capacity)
        base = score_targeting_scenario(
            p,
            count,
            customer_value=customer_value,
            offer_cost=offer_cost,
            offer_success_rate=offer_success_rate,
        )
        row = {
            "scenario_type": "assumption_based_retention_scenario",
            "target_fraction": float(fraction),
            "targeting_capacity": capacity,
            "budget": budget,
            "assumed_customer_value": customer_value,
            "assumed_offer_cost": offer_cost,
            "assumed_offer_success_rate": offer_success_rate,
            **{k: v for k, v in base.items() if k != "defaults_are_illustrative"},
            "not_causal_uplift": True,
        }
        if (
            labels is not None
            and decisions is not None
            and false_positive_cost is not None
            and false_negative_cost is not None
        ):
            y = np.asarray(labels)
            d = np.asarray(decisions)
            if len(y) != len(p) or len(d) != len(p):
                raise ValueError("labels/decisions must match probabilities length")
            selected = order[:count]
            fp = int(((d[selected] == 1) & (y[selected] == 0)).sum()) if count else 0
            fn = int(((d[selected] == 0) & (y[selected] == 1)).sum()) if count else 0
            decision_cost = fp * false_positive_cost + fn * false_negative_cost
            row["illustrative_false_positives"] = fp
            row["illustrative_false_negatives"] = fn
            row["illustrative_decision_cost"] = decision_cost
            row["scenario_net_value_after_decision_cost"] = (
                row["scenario_net_value"] - decision_cost
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_analysis(
    probabilities,
    *,
    base_fraction: float = 0.20,
    customer_value: float = DEFAULTS["customer_value"],
    offer_cost: float = DEFAULTS["offer_cost"],
    offer_success_rate: float = DEFAULTS["offer_success_rate"],
    max_customers: int | None = None,
    budget: float | None = None,
    success_rates: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    customer_values: tuple[float, ...] = (250.0, 500.0, 750.0, 1000.0),
    offer_costs: tuple[float, ...] = (10.0, 25.0, 50.0, 75.0),
) -> pd.DataFrame:
    """Vary one assumption at a time; values are scenario assumptions, not causal effects."""
    p = _validate_inputs(
        probabilities,
        customer_value=customer_value,
        offer_cost=offer_cost,
        offer_success_rate=offer_success_rate,
        max_customers=max_customers,
        budget=budget,
        false_positive_cost=None,
        false_negative_cost=None,
    )
    n = len(p)
    capacity = _capacity(
        n, max_customers=max_customers, budget=budget, offer_cost=offer_cost
    )
    count = 0 if n == 0 or capacity == 0 else min(
        max(1, int(np.ceil(n * base_fraction))), capacity
    )
    rows = []
    for rate in success_rates:
        s = score_targeting_scenario(
            p,
            count,
            customer_value=customer_value,
            offer_cost=offer_cost,
            offer_success_rate=rate,
        )
        rows.append(
            {
                "varied_parameter": "assumed_offer_success_rate",
                "varied_value": rate,
                "base_fraction": base_fraction,
                "customers_targeted": s["customers_targeted"],
                "scenario_net_value": s["scenario_net_value"],
                "assumed_customers_retained": s["assumed_customers_retained"],
                "not_causal_uplift": True,
            }
        )
    for value in customer_values:
        s = score_targeting_scenario(
            p,
            count,
            customer_value=value,
            offer_cost=offer_cost,
            offer_success_rate=offer_success_rate,
        )
        rows.append(
            {
                "varied_parameter": "assumed_customer_value",
                "varied_value": value,
                "base_fraction": base_fraction,
                "customers_targeted": s["customers_targeted"],
                "scenario_net_value": s["scenario_net_value"],
                "assumed_customers_retained": s["assumed_customers_retained"],
                "not_causal_uplift": True,
            }
        )
    for cost in offer_costs:
        # Recompute capacity if budget constrains affordability at this offer cost.
        cap = _capacity(n, max_customers=max_customers, budget=budget, offer_cost=cost)
        ccount = 0 if n == 0 or cap == 0 else min(max(1, int(np.ceil(n * base_fraction))), cap)
        s = score_targeting_scenario(
            p,
            ccount,
            customer_value=customer_value,
            offer_cost=cost,
            offer_success_rate=offer_success_rate,
        )
        rows.append(
            {
                "varied_parameter": "assumed_offer_cost",
                "varied_value": cost,
                "base_fraction": base_fraction,
                "customers_targeted": s["customers_targeted"],
                "scenario_net_value": s["scenario_net_value"],
                "assumed_customers_retained": s["assumed_customers_retained"],
                "not_causal_uplift": True,
            }
        )
    return pd.DataFrame(rows)


def write_scenario_artifacts(
    probabilities,
    out_dir: str | Path,
    *,
    customer_value: float = DEFAULTS["customer_value"],
    offer_cost: float = DEFAULTS["offer_cost"],
    offer_success_rate: float = DEFAULTS["offer_success_rate"],
    max_customers: int | None = None,
    budget: float | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = campaign_scenarios(
        probabilities,
        customer_value=customer_value,
        offer_cost=offer_cost,
        offer_success_rate=offer_success_rate,
        max_customers=max_customers,
        budget=budget,
    )
    sensitivity = sensitivity_analysis(
        probabilities,
        customer_value=customer_value,
        offer_cost=offer_cost,
        offer_success_rate=offer_success_rate,
        max_customers=max_customers,
        budget=budget,
    )
    scenario_path = out / "campaign_scenarios.csv"
    sensitivity_path = out / "campaign_sensitivity.csv"
    assumptions_path = out / "campaign_assumptions.json"
    scenarios.to_csv(scenario_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    pd.Series(
        {
            "assumption_note": ASSUMPTION_NOTE,
            "customer_value": customer_value,
            "offer_cost": offer_cost,
            "offer_success_rate": offer_success_rate,
            "max_customers": max_customers,
            "budget": budget,
            "defaults_are_illustrative": True,
            "not_causal_uplift": True,
        }
    ).to_json(assumptions_path)
    return {
        "campaign_scenarios": scenario_path,
        "campaign_sensitivity": sensitivity_path,
        "campaign_assumptions": assumptions_path,
    }
