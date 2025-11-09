import re
import numpy as np
import pandas as pd
from utils.date_utils import parse_any_date

def to_num_safe(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^0-9\.\-]", "", str(x))
    try: return float(s) if s != "" else np.nan
    except: return np.nan

def parse_credit_history_to_months(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    y = re.search(r"(\d+)\s*year", s)
    m = re.search(r"(\d+)\s*month", s)
    return (int(y.group(1)) if y else 0)*12 + (int(m.group(1)) if m else 0)

def _to_num_series(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9\.\-]","",regex=True), errors="coerce")

def clean_attributes_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    for col in ("customer_id","name","age","ssn","occupation","snapshot_date"):
        if col not in x.columns: x[col] = np.nan
    x = x.drop(columns=["name","ssn"], errors="ignore")
    x["occupation"] = (x["occupation"].astype(str).str.strip()
                       .replace({"_":"Unknown"}).replace("", "Unknown"))
    x["age"] = _to_num_series(x["age"])
    med = x["age"].median(skipna=True)
    x.loc[(x["age"]<18) | (x["age"]>74) | (x["age"].isna()), "age"] = med
    x["customer_id"] = x["customer_id"].astype(str).str.strip()
    x["snapshot_date"] = parse_any_date(x["snapshot_date"])
    return x

def clean_financials_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    x["customer_id"] = x["customer_id"].astype(str).str.strip()
    x["snapshot_date"] = parse_any_date(x.get("snapshot_date"))
    num_fix = [
        "annual_income","monthly_inhand_salary","num_bank_accounts","num_credit_card",
        "interest_rate","num_of_loan","delay_from_due_date","num_of_delayed_payment",
        "changed_credit_limit","num_credit_inquiries","outstanding_debt",
        "credit_utilization_ratio","total_emi_per_month","amount_invested_monthly","monthly_balance"
    ]
    for c in num_fix:
        if c in x.columns: x[c] = _to_num_series(x[c])

    repl = {
        "num_bank_accounts": [-1],
        "num_of_loan": [-100],
        "delay_from_due_date": [-1,-2,-3,-4,-5],
        "num_of_delayed_payment": [-3],
    }
    for c,bads in repl.items():
        if c in x.columns:
            med = x[c].replace(bads, np.nan).median(skipna=True)
            x[c] = x[c].replace(bads, np.nan).fillna(med)

    if "credit_mix" in x.columns:
        x["credit_mix"] = x["credit_mix"].replace({"_":"Unknown"}).fillna("Unknown")
    if "payment_of_min_amount" in x.columns:
        x["payment_of_min_amount"] = x["payment_of_min_amount"].replace({"NM":"No","nm":"No"}).fillna("Unknown")
    if "payment_behaviour" in x.columns:
        x["payment_behaviour"] = x["payment_behaviour"].replace({"!@9#%8":"Unknown"}).fillna("Unknown")
    if "credit_history_age" in x.columns:
        x["credit_history_months"] = x["credit_history_age"].apply(parse_credit_history_to_months)
    return x

def clean_clickstream_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    x["customer_id"] = x["customer_id"].astype(str).str.strip()
    x["snapshot_date"] = parse_any_date(x.get("snapshot_date"))
    fe_cols = [c for c in x.columns if c.startswith("fe_")]
    for c in fe_cols: x[c] = pd.to_numeric(x[c], errors="coerce")
    return x