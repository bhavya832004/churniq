import numpy as np
import pandas as pd

def campaign_scenarios(probabilities, customer_value=500.0, offer_cost=25.0,
                       offer_success_rate=0.20, max_customers=None) -> pd.DataFrame:
    """Illustrative scenario calculator; not a causal uplift estimator."""
    p = np.clip(np.asarray(probabilities, dtype=float), 0, 1)
    if customer_value < 0 or offer_cost < 0:
        raise ValueError("customer_value and offer_cost must be non-negative")
    if not 0 <= offer_success_rate <= 1:
        raise ValueError("offer_success_rate must be between 0 and 1")
    order = np.argsort(-p)
    rows = []
    n = len(p)
    for fraction in [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
        count = min(max(1, int(np.ceil(n * fraction))), max_customers or n) if n else 0
        selected = order[:count]
        expected_churners = float(p[selected].sum())
        expected_saved = expected_churners * offer_success_rate
        cost = count * offer_cost
        value = expected_saved * customer_value
        rows.append({"target_fraction": fraction, "customers_targeted": count,
          "mean_churn_probability": float(p[selected].mean()) if count else 0.0,
          "expected_churners_without_offer": expected_churners,
          "assumed_customers_retained": expected_saved, "campaign_cost": cost,
          "expected_retained_value": value, "scenario_net_value": value-cost})
    return pd.DataFrame(rows)
