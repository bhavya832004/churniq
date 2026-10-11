# Methodology

- Target: `Churn = Yes`; `customerID` is excluded from predictors.
- The dataset is a static snapshot, so this starter is classification practice rather than time-aware churn forecasting.
- Stratified train/validation/test split.
- Preprocessing is fitted inside each pipeline using training data only.
- Model selection uses validation PR-AUC.
- Operating threshold is selected on validation data using F2 to emphasize recall.
- Final metrics are calculated on the held-out test set.

## Metrics
- PR-AUC / average precision: positive-class ranking quality.
- ROC-AUC: ranking quality across thresholds.
- Precision and recall: intervention workload versus churner coverage.
- F1: precision/recall harmonic mean.
- Brier score and calibration curve: probability quality.
- Lift by decile: observed churn rate by ranked risk groups.

Permutation importance measures how model score changes when a feature is permuted. It does not establish causality. The business calculator is illustrative; causal incremental retention requires appropriate treatment/control data.
