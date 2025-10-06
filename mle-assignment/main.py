# main.py
# =======  CS611 MLE A1 — Monthly Backfill, Medallion, Time-Safe FE/Labels  =======
# - Bronze:  CSV → Parquet (lowercase cols)
# - Silver:  monthly snapshot_date loop; loan_daily(T) = LMS history ≤ T
# - Gold:    features(T) use attrs/fins/click ≤ T (last-known by customer),
#            labels(T) = 1 if exists overdue_amt>0 in (T, T+MOB] OR leftover balance at final installment
# - Run:
#     python main.py                         # run all stages
#     python main.py --stage silver          # run only silver
#     python main.py --start 2023-05-01 --end 2023-11-01  # limit months for quick test
# ================================================================================

import os, re, argparse, glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

# ---------------- Paths & Params ----------------
DATA_DIR = "data"                 # raw CSV folder
BRONZE_DIR = "datamart/bronze"
SILVER_DIR = "datamart/silver"
GOLD_DIR   = "datamart/gold"

RAW_FILES = {
    "lms":         f"{DATA_DIR}/lms_loan_daily.csv",
    # accept either file name (your sample used 'features_financial' header; we normalize)
    "financials":  f"{DATA_DIR}/features_financials.csv",
    "attributes":  f"{DATA_DIR}/features_attributes.csv",
    "clickstream": f"{DATA_DIR}/feature_clickstream.csv",
}

# Default backfill window (override via CLI flags)
START = "2023-01-01"
END   = "2024-12-01"
MOB_MONTHS = 6  # look-ahead window months for label

# ---------------- Utils ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def month_starts(start_str: str, end_str: str):
    d = datetime.fromisoformat(start_str).replace(day=1)
    end = datetime.fromisoformat(end_str)
    out = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d = (d + relativedelta(months=1)).replace(day=1)
    return out

def to_num_safe(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[^0-9\.\-]", "", s)  # remove non-numeric except dot/minus
    try:
        return float(s) if s != "" else np.nan
    except Exception:
        return np.nan

def parse_credit_history_to_months(s):
    # "10 Years and 9 Months" -> 129 months
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    y = re.search(r"(\d+)\s*year", s)
    m = re.search(r"(\d+)\s*month", s)
    years = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    return years * 12 + months

def lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def parse_any_date(series: pd.Series) -> pd.Series:
    # thử m/d/yy
    s = pd.to_datetime(series, errors="coerce", format="%m/%d/%y")
    mask = s.isna() & series.notna()
    if mask.any():
        # thử m/d/yyyy
        s2 = pd.to_datetime(series[mask], errors="coerce", format="%m/%d/%Y")
        s.loc[mask] = s2
    mask2 = s.isna() & series.notna()
    if mask2.any():
        # cuối cùng: để pandas suy đoán
        s3 = pd.to_datetime(series[mask2])
        s.loc[mask2] = s3
    return s

# ---------------- BRONZE ----------------
def bronze_ingest():
    print("=== BRONZE: ingest raw CSV -> parquet (lowercase cols) ===")
    ensure_dir(BRONZE_DIR)

    # handle optional alternate filename for financials
    if not os.path.exists(RAW_FILES["financials"]):
        alt = f"{DATA_DIR}/features_financial.csv"
        if os.path.exists(alt):
            RAW_FILES["financials"] = alt

    for name, path in RAW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN][BRONZE] Missing raw file: {path}")
            continue
        df = pd.read_csv(path)
        df = lower_columns(df)
        df.to_parquet(f"{BRONZE_DIR}/{name}.parquet", index=False)
        print(f"[BRONZE] {name} -> {BRONZE_DIR}/{name}.parquet rows={len(df)}")
    print("✅ Bronze done\n")

# ---------------- SILVER ----------------
def silver_backfill(start_str=START, end_str=END):
    print("=== SILVER: backfill monthly loan_daily(T) = LMS history ≤ T ===")
    ensure_dir(f"{SILVER_DIR}/loan_daily")

    # load bronze once
    lms_path = f"{BRONZE_DIR}/lms.parquet"
    if not os.path.exists(lms_path):
        raise FileNotFoundError("Bronze lms parquet not found. Run bronze first.")

    lms = pd.read_parquet(lms_path)
    # normalize date columns for LMS
    # use snapshot_date from raw as the 'post_date' line-of-business date
    # also parse loan_start_date if present
    if "snapshot_date" in lms.columns:
        lms["post_date"] = parse_any_date(lms["snapshot_date"])
    else:
        # fallback: pick first *date-like* column
        date_cols = [c for c in lms.columns if "date" in c]
        if date_cols:
            lms["post_date"] = pd.to_datetime(lms[date_cols[0]], errors="coerce")
        else:
            lms["post_date"] = pd.NaT

    if "loan_start_date" in lms.columns:
        lms["loan_start_date"] = parse_any_date(lms["loan_start_date"])

    # basic cleaning
    lms = lms.drop_duplicates().copy()

    # create monthly partitions
    for T in month_starts(start_str, end_str):
        T_dt = pd.to_datetime(T)
        out_dir = f"{SILVER_DIR}/loan_daily/snapshot_date={T}"
        ensure_dir(out_dir)

        # base mask: post_date ≤ T (nếu có)
                # chỉ giữ lịch sử ≤ T:
        # - nếu có post_date: dùng post_date ≤ T
        # - nếu post_date bị NaT: fallback loan_start_date ≤ T
        post = lms["post_date"] if "post_date" in lms.columns else pd.NaT
        start = lms["loan_start_date"] if "loan_start_date" in lms.columns else pd.NaT

        mask = pd.Series(False, index=lms.index)
        if "post_date" in lms.columns:
            mask = mask | (post.notna() & (post <= T_dt))
        if "loan_start_date" in lms.columns:
            mask = mask | (post.isna() & start.notna() & (start <= T_dt))

        lms_T = lms[mask].copy()
        lms_T.to_parquet(f"{out_dir}/loan_daily.parquet", index=False)
        print(f"[SILVER] {T} loan_daily rows={len(lms_T)} -> {out_dir}/loan_daily.parquet")
    print("✅ Silver done\n")

# ---------------- GOLD HELPERS ----------------
def load_silver_sources():
    # We keep cleaned "sources" from bronze; we'll clean them here for gold use
    paths = {
        "click":  f"{BRONZE_DIR}/clickstream.parquet",
        "attrs":  f"{BRONZE_DIR}/attributes.parquet",
        "fins":   f"{BRONZE_DIR}/financials.parquet",
        "lms":    f"{BRONZE_DIR}/lms.parquet",
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing bronze parquet for {k}: {p}")

    click = pd.read_parquet(paths["click"])
    attrs = pd.read_parquet(paths["attrs"])
    fins  = pd.read_parquet(paths["fins"])
    lms   = pd.read_parquet(paths["lms"])

    # date parsing for non-LMS sources (your sample uses m/d/yy)
    for df in (attrs, fins, click):
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = parse_any_date(df["snapshot_date"])

    # financials cleaning per your sample
    num_cols = [
        "annual_income","monthly_inhand_salary","num_bank_accounts","num_credit_card",
        "interest_rate","num_of_loan","delay_from_due_date","num_of_delayed_payment",
        "changed_credit_limit","num_credit_inquiries","outstanding_debt",
        "credit_utilization_ratio","total_emi_per_month","amount_invested_monthly","monthly_balance"
    ]
    for c in num_cols:
        if c in fins.columns:
            fins[c] = fins[c].apply(to_num_safe)

    if "credit_history_age" in fins.columns:
        fins["credit_history_months"] = fins["credit_history_age"].apply(parse_credit_history_to_months)

    if "credit_mix" in fins.columns:
        fins["credit_mix"] = fins["credit_mix"].replace({None:"unknown", "_":"unknown"}).fillna("unknown")
    if "payment_of_min_amount" in fins.columns:
        fins["payment_of_min_amount"] = fins["payment_of_min_amount"].replace({"nm":"no","NM":"no"}).fillna("unknown")

    # LMS dates again for the global (all-time) view used by labels
    if "snapshot_date" in lms.columns:
        lms["post_date"] = parse_any_date(lms["snapshot_date"])
    else:
        date_cols = [c for c in lms.columns if "date" in c]
        if date_cols:
            lms["post_date"] = pd.to_datetime(lms[date_cols[0]], errors="coerce")
        else:
            lms["post_date"] = pd.NaT

    if "loan_start_date" in lms.columns:
        lms["loan_start_date"] = parse_any_date(lms["loan_start_date"])

    # harmonize dtype basics
    for df in (click, attrs, fins, lms):
        # standardize id columns to str
        for idc in ("customer_id","loan_id"):
            if idc in df.columns:
                df[idc] = df[idc].astype(str).str.strip()

    return click, attrs, fins, lms

def last_known_by_customer(df: pd.DataFrame, T_dt: pd.Timestamp) -> pd.DataFrame:
    """Take last row ≤ T_dt per customer_id using snapshot_date if exists; else use all-time last."""
    if "customer_id" not in df.columns:
        return pd.DataFrame(columns=["customer_id"])  # no-op
    x = df.copy()
    if "snapshot_date" in x.columns:
        x = x[x["snapshot_date"].notna()]
        x = x[x["snapshot_date"] <= T_dt]
        x = x.sort_values(["customer_id","snapshot_date"]).groupby("customer_id").tail(1)
    else:
        # no time field -> assume timeless; keep latest appearance per customer
        x = x.drop_duplicates(subset=["customer_id"], keep="last")
    return x

def make_labels_for_T_using_overdue(lms_all: pd.DataFrame, T_dt: pd.Timestamp, mob: int) -> pd.DataFrame:
    """
    Label(T, loan_id) = 1 if exists overdue_amt>0 in (T, T+mob] OR
                        leftover balance>0 at final installment (any time).
    """
    end_dt = (T_dt + relativedelta(months=mob))

    need = {"loan_id","post_date","overdue_amt","installment_num","tenure","balance"}
    missing = [c for c in need if c not in lms_all.columns]
    if missing:
        raise ValueError(f"Missing columns in lms: {missing}")

    lms_all = lms_all.copy()
    lms_all["overdue_amt"] = lms_all["overdue_amt"].apply(to_num_safe)
    lms_all["balance"] = lms_all["balance"].apply(to_num_safe)

    # overdue in look-ahead window
    future = lms_all[(lms_all["post_date"] > T_dt) & (lms_all["post_date"] <= end_dt)].copy()
    overdue_any = (
        future.assign(overdue_flag = future["overdue_amt"].fillna(0).gt(0).astype(int))
              .groupby("loan_id", as_index=False)["overdue_flag"].max()
    )

    # leftover balance at final installment (global)
    last_inst = lms_all[lms_all["installment_num"] == lms_all["tenure"]].copy()
    leftover = (
        last_inst.assign(leftover_flag = last_inst["balance"].fillna(0).gt(0).astype(int))
                 .groupby("loan_id", as_index=False)["leftover_flag"].max()
    )

    labels = pd.merge(overdue_any, leftover, on="loan_id", how="outer").fillna(0)
    labels["label_default"] = ((labels["overdue_flag"] > 0) | (labels["leftover_flag"] > 0)).astype(int)
    return labels[["loan_id","label_default"]]

def build_features_for_T(T_dt: pd.Timestamp, lms_hist: pd.DataFrame,
                         attrs: pd.DataFrame, fins: pd.DataFrame, click: pd.DataFrame) -> pd.DataFrame:
    """
    Feature snapshot at T: base on loans observed up to T, merged with
    last-known attributes/financials/click (each filtered to ≤ T).
    """
    if lms_hist.empty:
        return pd.DataFrame()

    # base loans at/≤T: one row per loan_id (and keep customer_id if present)
    key_cols = [c for c in ["loan_id","customer_id","installment_num","tenure","loan_start_date","post_date","loan_amt","balance","overdue_amt","due_amt","paid_amt"] if c in lms_hist.columns]
    loans = (lms_hist.sort_values(["loan_id","post_date"])
                    .groupby("loan_id")
                    .tail(1))[key_cols].copy()

    # last-known per customer ≤ T
    attrs_T = last_known_by_customer(attrs, T_dt)
    fins_T  = last_known_by_customer(fins,  T_dt)
    click_T = last_known_by_customer(click, T_dt)

    feats = loans.copy()
    if "customer_id" in feats.columns:
        if "customer_id" in attrs_T.columns:
            feats = feats.merge(attrs_T.drop(columns=["snapshot_date"], errors="ignore"),
                                on="customer_id", how="left", suffixes=(None,"_attrs"))
        if "customer_id" in fins_T.columns:
            feats = feats.merge(fins_T.drop(columns=["snapshot_date","credit_history_age"], errors="ignore"),
                                on="customer_id", how="left", suffixes=(None,"_fins"))
        if "customer_id" in click_T.columns:
            feats = feats.merge(click_T.drop(columns=["snapshot_date"], errors="ignore"),
                                on="customer_id", how="left", suffixes=(None,"_click"))

    return feats

# ---------------- GOLD ----------------
def gold_backfill(start_str=START, end_str=END, mob_months=MOB_MONTHS):
    print("=== GOLD: features/labels monthly (partitioned by snapshot_date) ===")
    ensure_dir(f"{GOLD_DIR}/feature_store")
    ensure_dir(f"{GOLD_DIR}/label_store")

    # load sources
    click, attrs, fins, lms_all = load_silver_sources()

    # loop snapshots
    for T in month_starts(start_str, end_str):
        T_dt = pd.to_datetime(T)
        # load lms history at T from silver
        lms_hist_path = f"{SILVER_DIR}/loan_daily/snapshot_date={T}/loan_daily.parquet"
        if not os.path.exists(lms_hist_path):
            print(f"[WARN][GOLD] Missing silver partition for {T}, skipping.")
            continue
        lms_hist = pd.read_parquet(lms_hist_path)

        # labels: (T, T+MOB]
        labels_T = make_labels_for_T_using_overdue(lms_all, T_dt, mob=mob_months).copy()
        labels_T["snapshot_date"] = T_dt
        out_lbl = f"{GOLD_DIR}/label_store/snapshot_date={T}"
        ensure_dir(out_lbl)
        labels_T.to_parquet(f"{out_lbl}/labels.parquet", index=False)

        # features: last-known ≤ T
        features_T = build_features_for_T(T_dt, lms_hist, attrs, fins, click).copy()
        features_T["snapshot_date"] = T_dt
        out_ftr = f"{GOLD_DIR}/feature_store/snapshot_date={T}"
        ensure_dir(out_ftr)
        features_T.to_parquet(f"{out_ftr}/features.parquet", index=False)

        print(f"[GOLD] {T} -> features={len(features_T)}  labels={len(labels_T)}")
    print("✅ Gold done\n")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bronze","silver","gold","all"], default="all")
    ap.add_argument("--start", type=str, default=START)
    ap.add_argument("--end",   type=str, default=END)
    ap.add_argument("--mob",   type=int, default=MOB_MONTHS)
    args = ap.parse_args()

    # normalize to absolute yyyy-mm-01
    start = pd.to_datetime(args.start).strftime("%Y-%m-%d")
    end   = pd.to_datetime(args.end).strftime("%Y-%m-%d")

    if args.stage in ("bronze","all"):
        bronze_ingest()
    if args.stage in ("silver","all"):
        silver_backfill(start, end)
    if args.stage in ("gold","all"):
        gold_backfill(start, end, args.mob)

if __name__ == "__main__":
    main()