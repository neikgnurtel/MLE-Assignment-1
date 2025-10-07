import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def last_known_by_customer_from_silver(table_df: pd.DataFrame) -> pd.DataFrame:
    """Lấy last row per customer_id theo snapshot_date (đầu vào đã lọc ≤T)"""
    if table_df.empty or "customer_id" not in table_df.columns:
        return pd.DataFrame(columns=["customer_id"])
    x = table_df.copy()
    if "snapshot_date" in x.columns:
        x = x[x["snapshot_date"].notna()]
        x = x.sort_values(["customer_id","snapshot_date"]).groupby("customer_id").tail(1)
    else:
        x = x.drop_duplicates(subset=["customer_id"], keep="last")
    return x

def aggregate_click_median_from_silver(click_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate click fe_* theo median + cờ has_clickstream_data (đầu vào đã lọc ≤T)"""
    if click_df.empty or "customer_id" not in click_df.columns:
        out = pd.DataFrame(columns=["customer_id","has_clickstream_data"])
        return out
    fe_cols = [c for c in click_df.columns if c.startswith("fe_")]
    if not fe_cols:
        out = click_df.drop_duplicates("customer_id")[["customer_id"]].copy()
        out["has_clickstream_data"] = 0
        return out
    agg = click_df.groupby("customer_id")[fe_cols].median().reset_index()
    agg["has_clickstream_data"] = 1
    return agg

def make_lms_window_features(lms_hist: pd.DataFrame, T_dt: pd.Timestamp) -> pd.DataFrame:
    if lms_hist.empty: return pd.DataFrame(columns=["loan_id"])
    df = lms_hist.copy()
    for c in ["overdue_amt","due_amt","paid_amt","balance","installment_num","tenure","loan_amt"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    win6, win3 = T_dt - relativedelta(months=6), T_dt - relativedelta(months=3)
    df = df[df["post_date"].notna() & (df["post_date"] <= T_dt)]

    last = (df.sort_values(["loan_id","post_date"]).groupby("loan_id").tail(1)
            [["loan_id","installment_num","tenure","balance","overdue_amt","due_amt","paid_amt","loan_amt"]]
            .rename(columns={"installment_num":"inst_num_last","balance":"balance_last",
                             "overdue_amt":"overdue_last","due_amt":"due_last","paid_amt":"paid_last",
                             "loan_amt":"loan_amt_last"}))
    last["remaining_tenor"] = (last["tenure"] - last["inst_num_last"]).clip(lower=0)

    w6 = df[df["post_date"] > win6]
    agg6 = (w6.assign(ovf=(w6["overdue_amt"].fillna(0)>0).astype(int))
              .groupby("loan_id").agg(num_overdue_6m=("ovf","sum"),
                                      max_overdue_6m=("overdue_amt","max"),
                                      sum_due_6m=("due_amt","sum"),
                                      sum_paid_6m=("paid_amt","sum")))
    agg6["paid_to_due_6m"] = (agg6["sum_paid_6m"]/agg6["sum_due_6m"]).replace([np.inf,-np.inf], np.nan)

    w3 = df[df["post_date"] > win3]
    agg3 = (w3.assign(ovf=(w3["overdue_amt"].fillna(0)>0).astype(int))
              .groupby("loan_id").agg(num_overdue_3m=("ovf","sum"),
                                      max_overdue_3m=("overdue_amt","max"),
                                      sum_due_3m=("due_amt","sum"),
                                      sum_paid_3m=("paid_amt","sum")))
    agg3["paid_to_due_3m"] = (agg3["sum_paid_3m"]/agg3["sum_due_3m"]).replace([np.inf,-np.inf], np.nan)

    return last.set_index("loan_id").join(agg6, how="left").join(agg3, how="left").reset_index()

def add_derived_features(after_merge: pd.DataFrame) -> pd.DataFrame:
    x = after_merge.copy()
    def sdiv(a,b):
        b = b.replace(0,np.nan)
        return a / (b + 1e-6)

    if {"total_emi_per_month","monthly_inhand_salary"}.issubset(x.columns):
        x["emi_to_income"] = sdiv(pd.to_numeric(x["total_emi_per_month"], errors="coerce"),
                                  pd.to_numeric(x["monthly_inhand_salary"], errors="coerce"))
    if {"outstanding_debt","monthly_balance"}.issubset(x.columns):
        x["debt_to_balance"] = sdiv(pd.to_numeric(x["outstanding_debt"], errors="coerce"),
                                    pd.to_numeric(x["monthly_balance"], errors="coerce"))
    if {"num_of_delayed_payment","num_of_loan"}.issubset(x.columns):
        x["delay_ratio"] = sdiv(pd.to_numeric(x["num_of_delayed_payment"], errors="coerce"),
                                pd.to_numeric(x["num_of_loan"], errors="coerce"))

    if "has_clickstream_data" in x.columns:
        x["has_clickstream_data"] = x["has_clickstream_data"].fillna(0).astype(int)
    return x
