import argparse
from pathlib import Path
import json
import sys

import pandas as pd

from .data import load_dataset_with_report, DataValidationError
from .eda import run_eda
from .sql_analytics import run_sql_analytics
from .modeling import train_and_evaluate
from .business import campaign_scenarios


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_audit(data_path: str, out_path: str) -> int:
    path = Path(out_path)
    try:
        _df, report = load_dataset_with_report(data_path, raise_on_error=False)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _write_report(report, path)
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"  - {error}", file=sys.stderr)
        print(f"Saved data quality report (failed validation) to {path}", file=sys.stderr)
        return 1

    print(f"Saved data quality report to {path}")
    return 0


def run_all(data_path, out_dir) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        df, report = load_dataset_with_report(data_path, raise_on_error=True)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except DataValidationError as exc:
        _write_report(exc.report, out / "data_quality.json")
        print(str(exc), file=sys.stderr)
        print(f"Saved data quality report (failed validation) to {out / 'data_quality.json'}")
        return 1

    _write_report(report, out / "data_quality.json")
    if report["warnings"]:
        print("Data quality warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    run_eda(df, out)
    run_sql_analytics(df, out)
    result = train_and_evaluate(df, out)
    scores = pd.read_csv(out / "customer_risk_scores.csv")
    campaign_scenarios(scores["churn_probability"]).to_csv(
        out / "campaign_scenarios.csv", index=False
    )
    print(json.dumps(result, indent=2))
    print(f"Artifacts saved to: {out.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churniq")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("all", help="Run data audit, EDA, SQL and ML workflow")
    cmd.add_argument("--data", required=True)
    cmd.add_argument("--out", default="artifacts")
    audit = sub.add_parser("audit", help="Only validate and summarize the dataset")
    audit.add_argument("--data", required=True)
    audit.add_argument("--out", default="artifacts/data_quality.json")
    args = parser.parse_args(argv)

    if args.command == "all":
        raise SystemExit(run_all(args.data, args.out))
    raise SystemExit(run_audit(args.data, args.out))


if __name__ == "__main__":
    main()
