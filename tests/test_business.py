import pytest
from churniq.business import campaign_scenarios

def test_campaign_scenarios_shape():
    result = campaign_scenarios([0.1, 0.5, 0.9], customer_value=100, offer_cost=5, offer_success_rate=0.2)
    assert len(result) == 6
    assert "scenario_net_value" in result.columns
    assert result.iloc[0]["customers_targeted"] >= 1

def test_invalid_success_rate():
    with pytest.raises(ValueError):
        campaign_scenarios([0.2], offer_success_rate=2)
