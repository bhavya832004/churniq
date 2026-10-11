import argparse
from pathlib import Path
import json
import pandas as pd
from .data import load_dataset, data_quality_report
from .eda import run_eda
from .sql_analytics import run_sql_analytics
from .modeling import train_and_evaluate
from .business import campaign_scenarios

def run_all(data_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    (out / "data_quality.json").write_text(json.dumps(data_quality_report(df), indent=2), encoding="utf-8")
    run_eda(df, out); run_sql_analytics(df, out)
    result = train_and_evaluate(df, out)
    scores = pd.read_csv(out / "customer_risk_scores.csv")
    campaign_scenarios(scores["churn_probability"]).to_csv(out / "campaign_scenarios.csv", index=False)
    print(json.dumps(result, indent=2))
    print(f"Artifacts saved to: {out.resolve()}")

def main():
    parser = argparse.ArgumentParser(prog="churniq")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("all", help="Run data audit, EDA, SQL and ML workflow")
    cmd.add_argument("--data", required=True); cmd.add_argument("--out", default="artifacts")
    audit = sub.add_parser("audit", help="Only validate and summarize the dataset")
    audit.add_argument("--data", required=True); audit.add_argument("--out", default="artifacts/data_quality.json")
    args = parser.parse_args()
    if args.command == "all":
        run_all(args.data, args.out)
    else:
        df = load_dataset(args.data); path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data_quality_report(df), indent=2), encoding="utf-8")
        print(f"Saved data quality report to {path}")

if __name__ == "__main__":
    main()
