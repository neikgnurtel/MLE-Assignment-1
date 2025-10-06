import os
import pandas as pd

BRONZE_DIR = "datamart/bronze"
SILVER_DIR  = "datamart/silver"

RAW_FILES = {
    "clickstream": "data/feature_clickstream.csv",
    "attributes":  "data/features_attributes.csv",
    "financials":  "data/features_financials.csv",
    "lms":         "data/lms_loan_daily.csv"
}

def bronze_ingest():
    os.makedirs(BRONZE_DIR, exist_ok=True)
    missing = []
    for name, path in RAW_FILES.items():
        if not os.path.exists(path):
            missing.append(path); continue
        print(f"[BRONZE] Ingesting {name} from {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]  # chuẩn hóa tên cột
        df.to_parquet(f"{BRONZE_DIR}/{name}.parquet", index=False)
    if missing:
        print("[WARN] Missing raw files:", missing)
    print("✅ Bronze ingest completed")

def _coerce_dates(df: pd.DataFrame):
    # Tự phát hiện cột có 'date', 'dt', 'time' và ép kiểu datetime an toàn
    for col in df.columns:
        lc = col.lower()
        if any(tok in lc for tok in ["date", "dt", "time"]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _basic_clean(df: pd.DataFrame):
    # drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"    - dropped {before - len(df)} duplicates")

    # fillna cho numeric; strip cho string
    num_cols = df.select_dtypes(include="number").columns
    obj_cols = df.select_dtypes(include="object").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip()).replace({"nan": None})

    return df

def silver_process():
    os.makedirs(SILVER_DIR, exist_ok=True)
    if not os.path.isdir(BRONZE_DIR):
        raise RuntimeError("Bronze not found. Run bronze_ingest() first.")
    files = [f for f in os.listdir(BRONZE_DIR) if f.endswith(".parquet")]
    if not files:
        raise RuntimeError("No parquet in bronze/. Did bronze_ingest succeed?")

    for f in files:
        name = f.replace(".parquet", "")
        print(f"[SILVER] Processing {name}")
        df = pd.read_parquet(f"{BRONZE_DIR}/{f}")
        df = _coerce_dates(df)
        df = _basic_clean(df)
        df.to_parquet(f"{SILVER_DIR}/{name}.parquet", index=False)
        print(f"    -> saved {SILVER_DIR}/{name}.parquet")
    print("✅ Silver process completed")

def main():
    # Bạn có thể chạy từng tầng theo thứ tự; hiện gọi cả 2 để tiện test
    bronze_ingest()
    silver_process()

if __name__ == "__main__":
    main()

