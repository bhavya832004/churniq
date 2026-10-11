import pandas as pd
import pytest
from churniq.data import load_dataset, data_quality_report, make_features_target

def test_load_and_split_features(tmp_path):
    df = pd.DataFrame({"customerID":["a","b","c","d"], "tenure":[1,2,3,4],
        "TotalCharges":["10","20"," ","40"], "Churn":["Yes","No","No","Yes"]})
    path = tmp_path / "tiny.csv"; df.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded["TotalCharges"].isna().sum() == 1
    X, y = make_features_target(loaded)
    assert "Churn" not in X and "customerID" not in X
    assert y.tolist() == [1,0,0,1]

def test_missing_target_raises(tmp_path):
    path = tmp_path / "bad.csv"; pd.DataFrame({"x":[1,2]}).to_csv(path, index=False)
    with pytest.raises(ValueError): load_dataset(path)

def test_quality_counts():
    report = data_quality_report(pd.DataFrame({"customerID":["a","a"], "Churn":["Yes","No"]}))
    assert report["row_count"] == 2
    assert report["duplicate_customer_ids"] == 1
