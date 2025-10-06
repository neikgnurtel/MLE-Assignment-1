import os
import pandas as pd

BRONZE_DIR = "datamart/bronze"
SILVER_DIR = "datamart/silver"
GOLD_DIR   = "datamart/gold"

RAW_FILES = {
    "clickstream": "data/feature_clickstream.csv",
    "attributes":  "data/features_attributes.csv",
    "financials":  "data/features_financials.csv",
    "lms":         "data/lms_loan_daily.csv"
}

# ---------------- Bronze ----------------
def bronze_ingest():
    os.makedirs(BRONZE_DIR, exist_ok=True)
    for name, path in RAW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file {path}")
            continue
        print(f"[BRONZE] Ingesting {name}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df.to_parquet(f"{BRONZE_DIR}/{name}.parquet", index=False)
    print("✅ Bronze ingest completed")

# ---------------- Silver ----------------
def _coerce_dates(df):
    for col in df.columns:
        if any(tok in col.lower() for tok in ["date", "dt", "time"]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _basic_clean(df):
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include="number").columns
    obj_cols = df.select_dtypes(include="object").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip()).replace({"nan": None})
    return df

def silver_process():
    os.makedirs(SILVER_DIR, exist_ok=True)
    files = [f for f in os.listdir(BRONZE_DIR) if f.endswith(".parquet")]
    for f in files:
        name = f.replace(".parquet", "")
        print(f"[SILVER] Processing {name}")
        df = pd.read_parquet(f"{BRONZE_DIR}/{f}")
        df = _coerce_dates(df)
        df = _basic_clean(df)
        df.to_parquet(f"{SILVER_DIR}/{name}.parquet", index=False)
    print("✅ Silver process completed")

# ---------------- Gold ----------------
def make_label_store(lms):
    def derive_label(df):
        df = df.sort_values("installment_num")
        overdue = (df["overdue_amt"] > 0).astype(int)
        if overdue.rolling(2).sum().max() >= 2:
            return 1
        last = df[df["installment_num"] == df["tenure"]]
        if not last.empty and last["balance"].iloc[0] > 0:
            return 1
        return 0
    labels = lms.groupby("loan_id").apply(derive_label).reset_index()
    labels.columns = ["loan_id", "label_default"]
    return labels

def make_feature_store(lms, attrs, fins, click):
    loans = lms[lms["installment_num"] == 0].copy()
    loans = loans.merge(attrs.drop(columns=["snapshot_date"], errors="ignore"), on="customer_id", how="left")
    loans = loans.merge(fins.drop(columns=["snapshot_date"], errors="ignore"), on="customer_id", how="left")
    loans = loans.merge(click.drop(columns=["snapshot_date"], errors="ignore"), on="customer_id", how="left")
    return loans

def gold_process():
    os.makedirs(GOLD_DIR, exist_ok=True)
    click = pd.read_parquet(f"{SILVER_DIR}/clickstream.parquet")
    attrs = pd.read_parquet(f"{SILVER_DIR}/attributes.parquet")
    fins  = pd.read_parquet(f"{SILVER_DIR}/financials.parquet")
    lms   = pd.read_parquet(f"{SILVER_DIR}/lms.parquet")

    labels = make_label_store(lms)
    features = make_feature_store(lms, attrs, fins, click)

    features.to_parquet(f"{GOLD_DIR}/features.parquet", index=False)
    labels.to_parquet(f"{GOLD_DIR}/labels.parquet", index=False)

    print("✅ Gold process completed")

# ---------------- Main ----------------
def main():
    bronze_ingest()
    silver_process()
    gold_process()

if __name__ == "__main__":
    main()
