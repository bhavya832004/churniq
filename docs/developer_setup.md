# Developer setup

## Requirements

- Python 3.10 or newer
- Git
- Network access for the first dependency install and optional dataset download

## Windows PowerShell

```powershell
cd churniq
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

## Linux / macOS

```bash
cd churniq
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

## Dataset

```powershell
python scripts/download_dataset.py
```

Expected path: `data/raw/Telco-Customer-Churn.csv`.

If the download URL fails, obtain the IBM Telco Customer Churn CSV from the documented IBM repository and save it to that path. Do not substitute an unrelated file.

## Full local pipeline

```powershell
python -m churniq.cli audit --data data/raw/Telco-Customer-Churn.csv --out artifacts/data_quality.json
python -m churniq.cli all --data data/raw/Telco-Customer-Churn.csv --out artifacts
streamlit run app/streamlit_app.py
```

Generated CSVs, model files, plots, and SQLite databases under `artifacts/` are gitignored by default.

## Data quality audit

```powershell
python -m churniq.cli audit --data data/raw/Telco-Customer-Churn.csv --out artifacts/data_quality.json
```

The JSON report includes `ok`, `errors`, `warnings`, `metrics`, `source_path`, and `preprocessing_notes`. Fatal validation failures exit with code 1 after writing the report.

## Notes

- Raw data and trained models are not committed.
- Unit tests do not require the Telco CSV.
- The campaign calculator is a scenario tool, not causal uplift modeling.
