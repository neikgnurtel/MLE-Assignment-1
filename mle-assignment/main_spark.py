# spark_main.py — CS611 A1 (PySpark runner, local[*])
# Semantics khớp bản pandas: Bronze -> Silver -> Gold, monthly backfill, time-safe.

import os, argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

DATA_DIR   = "../data"
BRONZE_DIR = "../datamart/bronze"
SILVER_DIR = "../datamart/silver"
GOLD_DIR   = "../datamart/gold"

RAW_FILES = {
    "lms":         f"{DATA_DIR}/lms_loan_daily.csv",
    "financials":  f"{DATA_DIR}/features_financials.csv",
    "attributes":  f"{DATA_DIR}/features_attributes.csv",
    "clickstream": f"{DATA_DIR}/feature_clickstream.csv",
}

def month_starts(start, end):
    d = datetime.fromisoformat(start).replace(day=1)
    e = datetime.fromisoformat(end)
    out=[]
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d = (d + relativedelta(months=1)).replace(day=1)
    return out

def spark():
    return (SparkSession.builder
            .appName("cs611-a1")
            .master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())

# ---------------- BRONZE (Spark) ----------------
def bronze_ingest(spark):
    os.makedirs(BRONZE_DIR, exist_ok=True)
    for name, path in RAW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] missing raw: {path}")
            continue
        df = spark.read.option("header", True).csv(path)
        # lowercase columns
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip().lower())
        df.write.mode("overwrite").parquet(f"{BRONZE_DIR}/{name}")
        print(f"[BRONZE] {name} -> {BRONZE_DIR}/{name}")

# ---------------- SILVER (Spark) ----------------
def parse_any_date(col):
    # parse m/d/yy then m/d/yyyy, fallback to to_timestamp
    c1 = F.to_timestamp(col, "M/d/yy")
    c2 = F.when(c1.isNull(), F.to_timestamp(col, "M/d/yyyy")).otherwise(c1)
    c3 = F.when(c2.isNull(), F.to_timestamp(col)).otherwise(c2)
    return c3

def silver_backfill(spark, start, end):
    os.makedirs(f"{SILVER_DIR}/loan_daily", exist_ok=True)
    lms = spark.read.parquet(f"{BRONZE_DIR}/lms")
    lms = (lms
           .withColumn("post_date", parse_any_date(F.col("snapshot_date")))
           .withColumn("loan_start_date", parse_any_date(F.col("loan_start_date")))
           .dropDuplicates())

    for T in month_starts(start, end):
        Tlit = F.to_timestamp(F.lit(T))
        out_dir = f"{SILVER_DIR}/loan_daily/snapshot_date={T}"
        # mask: post_date ≤ T; fallback nếu post_date null thì loan_start_date ≤ T
        lms_T = (lms
                 .withColumn("_use_post", F.col("post_date").isNotNull())
                 .filter((F.col("_use_post") & (F.col("post_date") <= Tlit)) |
                         (~F.col("_use_post") & F.col("loan_start_date").isNotNull() &
                          (F.col("loan_start_date") <= Tlit)))
                 .drop("_use_post"))
        lms_T.write.mode("overwrite").parquet(f"{out_dir}/loan_daily.parquet")
        print(f"[SILVER] {T} rows={lms_T.count()} -> {out_dir}")

# ---------------- GOLD (Spark) ----------------
def last_known_on_or_before(df, Tlit):
    # keep last row ≤ T per customer_id using snapshot_date
    if "customer_id" not in df.columns or "snapshot_date" not in df.columns:
        return df.dropDuplicates(["customer_id"]) if "customer_id" in df.columns else df
    df2 = df.withColumn("snapshot_date_ts", parse_any_date(F.col("snapshot_date"))) \
            .filter(F.col("snapshot_date_ts").isNotNull() & (F.col("snapshot_date_ts") <= Tlit))
    w = Window.partitionBy("customer_id").orderBy(F.col("snapshot_date_ts").asc())
    return df2.withColumn("_rn", F.row_number().over(w)) \
              .filter(F.col("_rn") == F.max("_rn").over(Window.partitionBy("customer_id"))) \
              .drop("_rn","snapshot_date_ts")

def make_labels_T(spark, T, mob):
    Tlit = F.to_timestamp(F.lit(T))
    endlit = F.to_timestamp(F.lit((datetime.fromisoformat(T) + relativedelta(months=mob)).strftime("%Y-%m-%d")))
    lms_all = spark.read.parquet(f"{BRONZE_DIR}/lms") \
                .withColumn("post_date", parse_any_date(F.col("snapshot_date"))) \
                .withColumn("installment_num", F.col("installment_num").cast("int")) \
                .withColumn("tenure", F.col("tenure").cast("int")) \
                .withColumn("overdue_amt", F.col("overdue_amt").cast("double")) \
                .withColumn("balance", F.col("balance").cast("double"))

    future = lms_all.filter((F.col("post_date") > Tlit) & (F.col("post_date") <= endlit)) \
                    .withColumn("overdue_flag", (F.col("overdue_amt") > F.lit(0)).cast("int")) \
                    .groupBy("loan_id").agg(F.max("overdue_flag").alias("overdue_flag"))
    leftover = lms_all.filter(F.col("installment_num") == F.col("tenure")) \
                      .withColumn("leftover_flag", (F.col("balance") > F.lit(0)).cast("int")) \
                      .groupBy("loan_id").agg(F.max("leftover_flag").alias("leftover_flag"))
    labels = future.join(leftover, "loan_id", "outer").na.fill(0) \
                   .withColumn("label_default",
                               ( (F.col("overdue_flag") > 0) | (F.col("leftover_flag") > 0) ).cast("int")) \
                   .select("loan_id","label_default")
    return labels

def build_features_T(spark, T):
    Tlit = F.to_timestamp(F.lit(T))
    # silver history at T
    lms_hist = spark.read.parquet(f"{SILVER_DIR}/loan_daily/snapshot_date={T}/loan_daily.parquet")
    # base loans: trạng thái gần T theo post_date (fallback loan_start_date)
    lms_hist = lms_hist.withColumn("post_date", parse_any_date(F.col("post_date"))) \
                       .withColumn("loan_start_date", parse_any_date(F.col("loan_start_date")))
    # choose “latest” per loan_id by whichever date exists
    dtcol = F.when(F.col("post_date").isNotNull(), F.col("post_date")) \
             .otherwise(F.col("loan_start_date"))
    w_loan = Window.partitionBy("loan_id").orderBy(dtcol.asc())
    base = lms_hist.withColumn("_rn", F.row_number().over(w_loan)) \
                   .filter(F.col("_rn") == F.max("_rn").over(Window.partitionBy("loan_id"))) \
                   .drop("_rn")

    # sources last-known ≤ T
    attrs = spark.read.parquet(f"{BRONZE_DIR}/attributes")
    fins  = spark.read.parquet(f"{BRONZE_DIR}/financials")
    click = spark.read.parquet(f"{BRONZE_DIR}/clickstream")

    attrs_T = last_known_on_or_before(attrs, Tlit)
    fins_T  = last_known_on_or_before(fins,  Tlit)
    click_T = last_known_on_or_before(click, Tlit)

    feats = base
    for s in [("customer_id", attrs_T), ("customer_id", fins_T), ("customer_id", click_T)]:
        key, d = s
        if key in feats.columns and key in d.columns:
            d2 = d.drop("snapshot_date")
            feats = feats.join(d2, on=key, how="left")

    return feats

def gold_backfill(spark, start, end, mob):
    os.makedirs(f"{GOLD_DIR}/feature_store", exist_ok=True)
    os.makedirs(f"{GOLD_DIR}/label_store", exist_ok=True)

    for T in month_starts(start, end):
        feats = build_features_T(spark, T).withColumn("snapshot_date", F.lit(T))
        labels = make_labels_T(spark, T, mob).withColumn("snapshot_date", F.lit(T))

        out_f = f"{GOLD_DIR}/feature_store/snapshot_date={T}"
        out_l = f"{GOLD_DIR}/label_store/snapshot_date={T}"
        feats.write.mode("overwrite").parquet(f"{out_f}/features.parquet")
        labels.write.mode("overwrite").parquet(f"{out_l}/labels.parquet")
        print(f"[GOLD] {T} -> features={feats.count()} labels={labels.count()}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bronze","silver","gold","all"], default="all")
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end",   type=str, default="2024-12-01")
    ap.add_argument("--mob",   type=int, default=6)
    args = ap.parse_args()

    sp = spark()
    try:
        if args.stage in ("bronze","all"):
            bronze_ingest(sp)
        if args.stage in ("silver","all"):
            silver_backfill(sp, args.start, args.end)
        if args.stage in ("gold","all"):
            gold_backfill(sp, args.start, args.end, args.mob)
    finally:
        sp.stop()