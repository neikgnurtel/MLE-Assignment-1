import os
import pandas as pd

RAW_FILES = {
    "clickstream": "data/feature_clickstream.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
    "lms": "data/lms_loan_daily.csv"
}

def bronze_ingest():
    os.makedirs("datamart/bronze", exist_ok=True)
    for name, path in RAW_FILES.items():
        print(f"Ingesting {name} from {path}")
        df = pd.read_csv(path)
        # chuẩn hóa tên cột: lower case
        df.columns = [c.lower() for c in df.columns]
        out_path = f"datamart/bronze/{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {out_path}")

def main():
    print("Running Bronze Ingest...")
    bronze_ingest()
    print("✅ Bronze ingest completed")

if __name__ == "__main__":
    main()
