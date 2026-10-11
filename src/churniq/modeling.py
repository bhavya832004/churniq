from pathlib import Path
import json, platform, sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix, classification_report,
    brier_score_loss, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)])

def _metrics(y_true, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    result = {"threshold": float(threshold),
      "accuracy": float(accuracy_score(y_true, predictions)),
      "precision": float(precision_score(y_true, predictions, zero_division=0)),
      "recall": float(recall_score(y_true, predictions, zero_division=0)),
      "f1": float(f1_score(y_true, predictions, zero_division=0)),
      "pr_auc": float(average_precision_score(y_true, probabilities)),
      "brier_score": float(brier_score_loss(y_true, probabilities)),
      "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0,1]).tolist()}
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) == 2 else None
    return result

def choose_threshold(y_true, probabilities, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    b2 = beta ** 2
    denom = b2 * precision[:-1] + recall[:-1]
    fbeta = np.divide((1+b2)*precision[:-1]*recall[:-1], denom,
                      out=np.zeros_like(denom), where=denom != 0)
    return float(thresholds[int(np.argmax(fbeta))])

def _lift_table(y_true, probabilities, n_bins=10):
    frame = pd.DataFrame({"actual": np.asarray(y_true), "probability": np.asarray(probabilities)})
    bins = min(n_bins, len(frame))
    frame["risk_decile"] = pd.qcut(frame["probability"].rank(method="first"), q=bins,
                                   labels=False, duplicates="drop") + 1
    table = frame.groupby("risk_decile", observed=False).agg(
        customers=("actual","size"), mean_predicted_risk=("probability","mean"),
        observed_churn_rate=("actual","mean")).reset_index().sort_values("risk_decile", ascending=False)
    overall = frame["actual"].mean()
    table["lift_vs_average"] = table["observed_churn_rate"] / overall if overall else np.nan
    return table

def train_and_evaluate(df: pd.DataFrame, out_dir: str | Path, random_state=42) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=[c for c in ["Churn", "customerID"] if c in df]).copy()
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, random_state=random_state, stratify=y_hold)

    models = {
      "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
      "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    }
    fitted, rows = {}, []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        valid_probs = pipe.predict_proba(X_valid)[:,1]
        threshold = choose_threshold(y_valid, valid_probs, beta=2)
        test_probs = pipe.predict_proba(X_test)[:,1]
        metrics = _metrics(y_test, test_probs, threshold)
        metrics["model"] = name
        fitted[name] = {"pipeline": pipe, "threshold": threshold, "validation_pr_auc":
            average_precision_score(y_valid, valid_probs), "metrics": metrics}
        rows.append({k:v for k,v in metrics.items() if k != "confusion_matrix"})
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)

    # Model selection is based on validation PR-AUC, never test metrics.
    selected_name = max(fitted, key=lambda n: fitted[n]["validation_pr_auc"])
    selected = fitted[selected_name]; pipe = selected["pipeline"]; threshold = selected["threshold"]
    probs = pipe.predict_proba(X_test)[:,1]
    metrics = _metrics(y_test, probs, threshold)
    metrics.update({"selected_model": selected_name, "validation_threshold": threshold,
      "threshold_selection": "F2 maximization on validation set",
      "train_rows": int(len(y_train)), "validation_rows": int(len(y_valid)),
      "test_rows": int(len(y_test)), "positive_class": "Churn=Yes"})
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(classification_report(y_test, (probs >= threshold).astype(int),
      output_dict=True, zero_division=0)).T.to_csv(out / "classification_report.csv")

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5,4)); im = ax.imshow(cm)
    ax.set(title=f"Test confusion matrix — {selected_name}", xlabel="Predicted", ylabel="Actual",
           xticks=[0,1], yticks=[0,1])
    for (i,j), val in np.ndenumerate(cm): ax.text(j, i, str(val), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1], "--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label=selected_name)
    ax.set(title="Calibration curve (test set)", xlabel="Mean predicted probability",
           ylabel="Observed churn rate"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "calibration_curve.png", dpi=150); plt.close(fig)

    _lift_table(y_test, probs).to_csv(out / "lift_by_decile.csv", index=False)
    perm = permutation_importance(pipe, X_test, y_test, scoring="average_precision",
                                  n_repeats=5, random_state=random_state, n_jobs=-1)
    pd.DataFrame({"feature": X_test.columns, "importance_mean": perm.importances_mean,
      "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False).to_csv(
      out / "permutation_importance.csv", index=False)

    # Demo scores for all rows. Training-row scores are in-sample, not unbiased predictions.
    all_probs = pipe.predict_proba(X)[:,1]
    scored = df[[c for c in ["customerID","Churn"] if c in df]].copy()
    scored["churn_probability"] = all_probs
    scored["risk_tier"] = pd.cut(scored["churn_probability"], [-0.001,.25,.5,.75,1.001],
      labels=["Low","Moderate","High","Very high"]).astype(str)
    scored.to_csv(out / "customer_risk_scores.csv", index=False)
    joblib.dump({"pipeline":pipe, "threshold":threshold, "model_name":selected_name}, out / "model.joblib")
    (out / "run_metadata.json").write_text(json.dumps({
      "python":sys.version, "platform":platform.platform(), "random_state":random_state,
      "selected_model":selected_name, "feature_columns":list(X.columns),
      "split":{"train":len(X_train),"validation":len(X_valid),"test":len(X_test)},
      "note":"Threshold and model selection use validation data; test set is held out for evaluation."
    }, indent=2), encoding="utf-8")
    return {"metrics":metrics, "selected_model":selected_name, "threshold":threshold}
