"""SQLite analytics for churn segments and data checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sqlite3

# Named queries kept here so outputs can be recreated from the same SQL.
SQL_QUERIES: dict[str, str] = {
    "sql_validation_checks": """
        SELECT
          COUNT(*) AS row_count,
          SUM(CASE WHEN Churn IS NULL OR TRIM(CAST(Churn AS TEXT)) = '' THEN 1 ELSE 0 END)
            AS missing_churn_labels,
          SUM(CASE WHEN Churn NOT IN ('Yes', 'No') THEN 1 ELSE 0 END) AS invalid_churn_labels,
          SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS null_total_charges,
          COUNT(DISTINCT customerID) AS distinct_customer_ids,
          COUNT(*) - COUNT(DISTINCT customerID) AS duplicate_customer_id_rows
        FROM customers
    """,
    "sql_contract_analysis": """
        SELECT
          Contract,
          COUNT(*) AS customers,
          SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churners,
          AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate,
          AVG(MonthlyCharges) AS avg_monthly_charges,
          AVG(CASE WHEN TotalCharges IS NULL THEN NULL ELSE TotalCharges END)
            AS avg_total_charges_non_null
        FROM customers
        WHERE Contract IS NOT NULL
        GROUP BY Contract
        ORDER BY churn_rate DESC
    """,
    "sql_payment_analysis": """
        SELECT
          PaymentMethod,
          COUNT(*) AS customers,
          SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churners,
          AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
        FROM customers
        WHERE PaymentMethod IS NOT NULL
        GROUP BY PaymentMethod
        ORDER BY churn_rate DESC
    """,
    "sql_contract_churn_ranked": """
        WITH contract_stats AS (
          SELECT
            Contract,
            COUNT(*) AS customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churners,
            AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
          FROM customers
          WHERE Contract IS NOT NULL
          GROUP BY Contract
        )
        SELECT
          Contract,
          customers,
          churners,
          churn_rate,
          RANK() OVER (ORDER BY churn_rate DESC) AS churn_rate_rank,
          ROUND(customers * 1.0 / SUM(customers) OVER (), 6) AS share_of_customers
        FROM contract_stats
        ORDER BY churn_rate_rank
    """,
    "sql_tenure_cohorts": """
        SELECT
          CASE
            WHEN tenure IS NULL THEN 'unknown'
            WHEN tenure <= 6 THEN '0-6'
            WHEN tenure <= 12 THEN '7-12'
            WHEN tenure <= 24 THEN '13-24'
            WHEN tenure <= 48 THEN '25-48'
            ELSE '49+'
          END AS tenure_band,
          COUNT(*) AS customers,
          SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churners,
          AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate,
          AVG(MonthlyCharges) AS avg_monthly_charges
        FROM customers
        GROUP BY 1
        ORDER BY
          CASE
            WHEN tenure_band = '0-6' THEN 1
            WHEN tenure_band = '7-12' THEN 2
            WHEN tenure_band = '13-24' THEN 3
            WHEN tenure_band = '25-48' THEN 4
            WHEN tenure_band = '49+' THEN 5
            ELSE 6
          END
    """,
    "sql_null_total_charges_by_tenure": """
        SELECT
          CASE
            WHEN tenure IS NULL THEN 'unknown'
            WHEN tenure <= 6 THEN '0-6'
            WHEN tenure <= 12 THEN '7-12'
            WHEN tenure <= 24 THEN '13-24'
            WHEN tenure <= 48 THEN '25-48'
            ELSE '49+'
          END AS tenure_band,
          COUNT(*) AS customers,
          SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS null_total_charges,
          ROUND(
            SUM(CASE WHEN TotalCharges IS NULL THEN 1.0 ELSE 0.0 END) / COUNT(*),
            6
          ) AS null_total_charges_rate
        FROM customers
        GROUP BY 1
        ORDER BY null_total_charges DESC
    """,
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace(" ", "_") for c in out.columns]
    return out


def run_sql_analytics(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Load customers into SQLite, run named analyses, write CSV artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "churniq.sqlite"
    sql_df = _normalize_columns(df)

    # Persist query text for recreation outside Python.
    (out / "sql_queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{sql.strip()}\n" for name, sql in SQL_QUERIES.items()),
        encoding="utf-8",
    )

    with sqlite3.connect(db_path) as conn:
        sql_df.to_sql("customers", conn, if_exists="replace", index=False)
        available = set(sql_df.columns)
        for name, sql in SQL_QUERIES.items():
            required = {
                "sql_validation_checks": {"Churn", "customerID"},
                "sql_contract_analysis": {"Contract", "Churn", "MonthlyCharges"},
                "sql_payment_analysis": {"PaymentMethod", "Churn"},
                "sql_contract_churn_ranked": {"Contract", "Churn"},
                "sql_tenure_cohorts": {"tenure", "Churn", "MonthlyCharges"},
                "sql_null_total_charges_by_tenure": {"tenure", "TotalCharges"},
            }.get(name, set())
            # TotalCharges optional for validation query null count.
            if name == "sql_validation_checks" and "TotalCharges" not in available:
                sql = """
                    SELECT
                      COUNT(*) AS row_count,
                      SUM(CASE WHEN Churn IS NULL OR TRIM(CAST(Churn AS TEXT)) = '' THEN 1 ELSE 0 END)
                        AS missing_churn_labels,
                      SUM(CASE WHEN Churn NOT IN ('Yes', 'No') THEN 1 ELSE 0 END)
                        AS invalid_churn_labels,
                      NULL AS null_total_charges,
                      COUNT(DISTINCT customerID) AS distinct_customer_ids,
                      COUNT(*) - COUNT(DISTINCT customerID) AS duplicate_customer_id_rows
                    FROM customers
                """
                required = {"Churn", "customerID"}
            if not required.issubset(available):
                continue
            pd.read_sql_query(sql, conn).to_csv(out / f"{name}.csv", index=False)
    return db_path
