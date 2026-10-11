from pathlib import Path
import sqlite3
import pandas as pd

def run_sql_analytics(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    db_path = out / "churniq.sqlite"
    sql_df = df.copy()
    sql_df.columns = [str(c).replace(" ", "_") for c in sql_df.columns]
    with sqlite3.connect(db_path) as conn:
        sql_df.to_sql("customers", conn, if_exists="replace", index=False)
        if "Contract" in sql_df:
            pd.read_sql_query('''
                SELECT Contract, COUNT(*) AS customers,
                  AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate,
                  AVG(MonthlyCharges) AS avg_monthly_charges
                FROM customers GROUP BY Contract ORDER BY churn_rate DESC
            ''', conn).to_csv(out / "sql_contract_analysis.csv", index=False)
        if "PaymentMethod" in sql_df:
            pd.read_sql_query('''
                SELECT PaymentMethod, COUNT(*) AS customers,
                  AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
                FROM customers GROUP BY PaymentMethod ORDER BY churn_rate DESC
            ''', conn).to_csv(out / "sql_payment_analysis.csv", index=False)
    return db_path
