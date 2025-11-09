# main.py — CS611 A1: Bronze → Silver(clean+partition) → Gold(join+FE+labels)
from utils.io_utils import ensure_dir, lower_columns
from utils.date_utils import month_starts, month_floor, parse_any_date
from utils.clean_utils import (
    to_num_safe, parse_credit_history_to_months, _to_num_series,
    clean_attributes_df, clean_financials_df, clean_clickstream_df
)
from utils.feature_utils import (
    last_known_by_customer_from_silver, aggregate_click_median_from_silver,
    make_lms_window_features, add_derived_features
)
import os, re, argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

# ---------------- Paths & Params ----------------
DATA_DIR   = "data"
BRONZE_DIR = "datamart/bronze"
SILVER_DIR = "datamart/silver"
GOLD_DIR   = "datamart/gold"

RAW_FILES = {
    "lms":         f"{DATA_DIR}/lms_loan_daily.csv",
    "financials":  f"{DATA_DIR}/features_financials.csv",  # accept alt name in bronze_ingest()
    "attributes":  f"{DATA_DIR}/features_attributes.csv",
    "clickstream": f"{DATA_DIR}/feature_clickstream.csv",
}

START = "2023-01-01"
END   = "2024-12-01"
MOB_MONTHS = 6

# ================= BRONZE =================
def bronze_ingest():
    print("=== BRONZE: ingest raw CSV -> parquet (lowercase cols) ===")
    ensure_dir(BRONZE_DIR)
    if not os.path.exists(RAW_FILES["financials"]):
        alt = f"{DATA_DIR}/features_financial.csv"
        if os.path.exists(alt): RAW_FILES["financials"] = alt
    for name, path in RAW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN][BRONZE] Missing raw file: {path}"); continue
        df = lower_columns(pd.read_csv(path))
        df.to_parquet(f"{BRONZE_DIR}/{name}.parquet", index=False)
        print(f"[BRONZE] {name} -> {BRONZE_DIR}/{name}.parquet rows={len(df)}")
    print("✅ Bronze done\n")

# ================= SILVER =================
def silver_backfill(start_str=START, end_str=END):
    print("=== SILVER: backfill monthly + cleaned partitions ===")
    # ---- LMS → loan_daily(T) = history ≤ T
    ensure_dir(f"{SILVER_DIR}/loan_daily")
    lms_path = f"{BRONZE_DIR}/lms.parquet"
    if not os.path.exists(lms_path): raise FileNotFoundError("Run bronze first: lms missing.")
    lms = pd.read_parquet(lms_path)
    if "snapshot_date" in lms.columns: lms["post_date"] = parse_any_date(lms["snapshot_date"])
    else:
        date_cols = [c for c in lms.columns if "date" in c]
        lms["post_date"] = pd.to_datetime(lms[date_cols[0]], errors="coerce") if date_cols else pd.NaT
    if "loan_start_date" in lms.columns:
        lms["loan_start_date"] = parse_any_date(lms["loan_start_date"])
    lms = lms.drop_duplicates().copy()

    # ---- Cleaned Silver cho attributes/financials/clickstream (partition theo snapshot_month & filter theo window)
    attrs_bz = pd.read_parquet(f"{BRONZE_DIR}/attributes.parquet")
    fins_bz  = pd.read_parquet(f"{BRONZE_DIR}/financials.parquet")
    click_bz = pd.read_parquet(f"{BRONZE_DIR}/clickstream.parquet")

    attrs_s = clean_attributes_df(attrs_bz)
    fins_s  = clean_financials_df(fins_bz)
    click_s = clean_clickstream_df(click_bz)

    start_dt = pd.to_datetime(start_str)
    end_dt   = pd.to_datetime(end_str)

    for name, dfc in [("attributes", attrs_s), ("financials", fins_s), ("clickstream", click_s)]:
        out_root = f"{SILVER_DIR}/{name}"; ensure_dir(out_root)
        if "snapshot_date" not in dfc.columns:
            dfc.to_parquet(f"{out_root}/{name}.parquet", index=False)
            continue

        dfc = dfc[dfc["snapshot_date"].notna()].copy()
        dfc["snapshot_month"] = month_floor(dfc["snapshot_date"])
        mask = (dfc["snapshot_month"] >= month_floor(start_dt)) & (dfc["snapshot_month"] <= month_floor(end_dt))
        dfc = dfc[mask]

        for Tpart, g in dfc.groupby(dfc["snapshot_month"].dt.strftime("%Y-%m-%d")):
            part = f"{out_root}/snapshot_date={Tpart}"; ensure_dir(part)
            g.drop(columns=["snapshot_month"], errors="ignore").to_parquet(f"{part}/{name}.parquet", index=False)

    # ---- loan_daily partitions
    for T in month_starts(start_str, end_str):
        T_dt = pd.to_datetime(T)
        out_dir = f"{SILVER_DIR}/loan_daily/snapshot_date={T}"
        ensure_dir(out_dir)
        post = lms.get("post_date", pd.NaT)
        start = lms.get("loan_start_date", pd.NaT)
        mask = pd.Series(False, index=lms.index)
        if "post_date" in lms.columns: mask |= (post.notna() & (post <= T_dt))
        if "loan_start_date" in lms.columns: mask |= (post.isna() & start.notna() & (start <= T_dt))
        lms_T = lms[mask].copy()
        lms_T.to_parquet(f"{out_dir}/loan_daily.parquet", index=False)
        print(f"[SILVER] {T} loan_daily rows={len(lms_T)} -> {out_dir}/loan_daily.parquet")
    print(f"[SILVER] attributes: {len(attrs_s)} | financials: {len(fins_s)} | clickstream: {len(click_s)} | loan_daily: {len(lms)}")
    print("✅ Silver done\n")

# ================= GOLD HELPERS (read from SILVER) =================
def list_silver_partitions(root_dir: str):
    if not os.path.exists(root_dir): return []
    parts = []
    for name in os.listdir(root_dir):
        if name.startswith("snapshot_date="):
            try:
                dt = datetime.fromisoformat(name.split("=",1)[1])
                parts.append((name, dt))
            except Exception:
                pass
    return sorted(parts, key=lambda x: x[1])

def read_silver_upto(table: str, T_dt: pd.Timestamp) -> pd.DataFrame:
    """Đọc & concat tất cả partitions silver/<table>/snapshot_date=<=T"""
    root = f"{SILVER_DIR}/{table}"
    parts = list_silver_partitions(root)
    sel = [p for p,dt in parts if dt <= T_dt]
    if not sel:
        f = f"{root}/{table}.parquet"
        return pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()
    dfs = []
    for p in sel:
        path = f"{root}/{p}/{table}.parquet"
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def make_labels_for_T_using_overdue(lms_all: pd.DataFrame, T_dt: pd.Timestamp, mob: int) -> pd.DataFrame:
    end_dt = (T_dt + relativedelta(months=mob))
    need = {"loan_id","post_date","overdue_amt","installment_num","tenure","balance"}
    miss = [c for c in need if c not in lms_all.columns]
    if miss: raise ValueError(f"Missing columns in lms: {miss}")
    x = lms_all.copy()
    x["overdue_amt"] = pd.to_numeric(x["overdue_amt"], errors="coerce")
    x["balance"] = pd.to_numeric(x["balance"], errors="coerce")

    future = x[(x["post_date"] > T_dt) & (x["post_date"] <= end_dt)].copy()
    overdue_any = (future.assign(overdue_flag=future["overdue_amt"].fillna(0).gt(0).astype(int))
                        .groupby("loan_id", as_index=False)["overdue_flag"].max())

    last_inst = x[x["installment_num"] == x["tenure"]].copy()
    leftover  = (last_inst.assign(leftover_flag=last_inst["balance"].fillna(0).gt(0).astype(int))
                        .groupby("loan_id", as_index=False)["leftover_flag"].max())

    labels = pd.merge(overdue_any, leftover, on="loan_id", how="outer").fillna(0)
    labels["label_default"] = ((labels["overdue_flag"]>0) | (labels["leftover_flag"]>0)).astype(int)
    return labels[["loan_id","label_default"]]

# ================= GOLD BUILD (read SILVER only) =================
def build_features_for_T(T_dt: pd.Timestamp, lms_hist: pd.DataFrame) -> pd.DataFrame:
    """Đọc Silver attributes/financials/click ≤ T, last-known/aggregate rồi join vào loan snapshot tại T"""
    if lms_hist.empty: return pd.DataFrame()
    key_cols = [c for c in ["loan_id","customer_id","installment_num","tenure","loan_start_date",
                            "post_date","loan_amt","balance","overdue_amt","due_amt","paid_amt"]
                if c in lms_hist.columns]
    loans = (lms_hist.sort_values(["loan_id","post_date"]).groupby("loan_id").tail(1))[key_cols].copy()

    # đọc silver up-to-T
    attrs_upto = read_silver_upto("attributes", T_dt)
    fins_upto  = read_silver_upto("financials", T_dt)
    click_upto = read_silver_upto("clickstream", T_dt)

    # last-known / aggregate
    attrs_T = last_known_by_customer_from_silver(attrs_upto)
    fins_T  = last_known_by_customer_from_silver(fins_upto)
    click_T = aggregate_click_median_from_silver(click_upto)

    feats = loans.copy()
    if "customer_id" in feats.columns:
        if not attrs_T.empty and "customer_id" in attrs_T.columns:
            feats = feats.merge(attrs_T.drop(columns=["snapshot_date"], errors="ignore"),
                                on="customer_id", how="left", suffixes=(None,"_attrs"))
        if not fins_T.empty and "customer_id" in fins_T.columns:
            feats = feats.merge(fins_T.drop(columns=["snapshot_date","credit_history_age"], errors="ignore"),
                                on="customer_id", how="left", suffixes=(None,"_fins"))
        if not click_T.empty and "customer_id" in click_T.columns:
            feats = feats.merge(click_T, on="customer_id", how="left")

    # fill fe_* NaN -> 0 (khách chưa có click)
    fe_cols = [c for c in feats.columns if c.startswith("fe_")]
    if fe_cols: feats[fe_cols] = feats[fe_cols].fillna(0)

    # LMS window FE
    fe_lms = make_lms_window_features(lms_hist, T_dt)
    if not fe_lms.empty: feats = feats.merge(fe_lms, on="loan_id", how="left")

    feats = add_derived_features(feats)
    return feats

def gold_backfill(start_str=START, end_str=END, mob_months=MOB_MONTHS):
    print("=== GOLD: features/labels monthly (partitioned by snapshot_date) ===")
    ensure_dir(f"{GOLD_DIR}/feature_store"); ensure_dir(f"{GOLD_DIR}/label_store")

    # labels dùng toàn bộ LMS (bronze) + post_date đã parse
    lms_all = pd.read_parquet(f"{BRONZE_DIR}/lms.parquet")
    lms_all["post_date"] = parse_any_date(lms_all.get("snapshot_date"))
    if "loan_start_date" in lms_all.columns:
        lms_all["loan_start_date"] = parse_any_date(lms_all["loan_start_date"])

    for T in month_starts(start_str, end_str):
        T_dt = pd.to_datetime(T)

        # lms history ≤ T từ Silver
        lms_hist_path = f"{SILVER_DIR}/loan_daily/snapshot_date={T}/loan_daily.parquet"
        if not os.path.exists(lms_hist_path):
            print(f"[WARN][GOLD] Missing silver partition for {T}, skipping.")
            continue
        lms_hist = pd.read_parquet(lms_hist_path)

        # labels: (T, T+MOB]
        labels_T = make_labels_for_T_using_overdue(lms_all, T_dt, mob=mob_months).copy()
        labels_T["snapshot_date"] = T_dt
        out_lbl = f"{GOLD_DIR}/label_store/snapshot_date={T}"; ensure_dir(out_lbl)
        labels_T.to_parquet(f"{out_lbl}/labels.parquet", index=False)

        # features: chỉ đọc Silver ≤ T
        features_T = build_features_for_T(T_dt, lms_hist).copy()
        features_T["snapshot_date"] = T_dt
        out_ftr = f"{GOLD_DIR}/feature_store/snapshot_date={T}"; ensure_dir(out_ftr)
        features_T.to_parquet(f"{out_ftr}/features.parquet", index=False)

        print(f"[GOLD] {T} -> features={len(features_T)}  labels={len(labels_T)}")
    print("✅ Gold done\n")

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bronze","silver","gold","all"], default="all")
    ap.add_argument("--start", type=str, default=START)
    ap.add_argument("--end",   type=str, default=END)
    ap.add_argument("--mob",   type=int, default=MOB_MONTHS)
    args = ap.parse_args()
    start = pd.to_datetime(args.start).strftime("%Y-%m-%d")
    end   = pd.to_datetime(args.end).strftime("%Y-%m-%d")

    if args.stage in ("bronze","all"): bronze_ingest()
    if args.stage in ("silver","all"): silver_backfill(start, end)
    if args.stage in ("gold","all"):   gold_backfill(start, end, args.mob)

if __name__ == "__main__":
    main()