import pandas as pd

from churniq.dashboard import mask_customer_id, select_out_of_sample_scores


def test_mask_customer_id():
    masked = mask_customer_id(pd.Series(["ABCD-1234", "xy"]))
    assert masked.iloc[0].startswith("AB")
    assert "***" in masked.iloc[0]
    assert masked.iloc[1] == "***"


def test_select_out_of_sample_scores():
    df = pd.DataFrame(
        {
            "dataset_split": ["train", "validation", "test"],
            "score_method": ["in_sample", "out_of_sample", "out_of_sample"],
            "churn_probability": [0.9, 0.5, 0.4],
        }
    )
    oos = select_out_of_sample_scores(df)
    assert set(oos["dataset_split"]) == {"validation", "test"}
    assert (oos["score_method"] == "out_of_sample").all()
