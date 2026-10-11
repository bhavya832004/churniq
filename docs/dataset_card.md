# Dataset card: IBM Telco Customer Churn

**Intended use:** educational portfolio demonstration of a data science workflow.

**Content:** customer-level telecom attributes, service subscriptions, billing details, tenure, and churn label.

**Limitations:** common teaching benchmark; static snapshot; may not represent current customers; no treatment assignment or retention intervention outcomes for causal uplift estimation.

**Data handling:** downloaded CSV is stored locally and ignored by Git. Review the dataset source's terms and attribution requirements before redistributing it.

**Validation contract:** `churniq.validation` / `churniq.cli audit` require target column `Churn` with values `Yes`/`No`. Invalid labels, empty files, and unreadable CSVs are fatal errors. Rows are not silently dropped. `TotalCharges` is coerced to numeric with conversion missings retained and counted. Duplicate rows/IDs and one-class targets are warnings.
