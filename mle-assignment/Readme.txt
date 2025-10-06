# MLE Assignment 1 – Medallion Backfill

All deliverables live in [`mle-assignment/`](mle-assignment).  
Raw data expected in `data/`, outputs written to `datamart/`.

## Quick Start (Docker, recommended)

From the **repo root**:
```bash
docker compose -f mle-assignment/docker-compose.yaml build

# Pandas runner
docker compose -f mle-assignment/docker-compose.yaml up pandas

# PySpark runner
docker compose -f mle-assignment/docker-compose.yaml up spark

# Change date range at Runtime
docker compose -f mle-assignment/docker-compose.yaml run --rm pandas \
  python mle-assignment/main.py --start 2023-01-01 --end 2024-06-01

docker compose -f mle-assignment/docker-compose.yaml run --rm spark \
  python mle-assignment/main_spark.py --start 2023-01-01 --end 2024-06-01

# Expected outputsdatamart/
  bronze/{lms,financials,attributes,clickstream}.parquet
  silver/loan_daily/snapshot_date=YYYY-MM-01/loan_daily.parquet
  gold/{feature_store,label_store}/snapshot_date=YYYY-MM-01/{features,labels}.parquet

# Sanity Check
python - <<'PY'
import pandas as pd
T="2023-09-01"
fe=pd.read_parquet(f"datamart/gold/feature_store/snapshot_date={T}/features.parquet")
lb=pd.read_parquet(f"datamart/gold/label_store/snapshot_date={T}/labels.parquet")
print("features:", len(fe), "labels:", len(lb))
PY
