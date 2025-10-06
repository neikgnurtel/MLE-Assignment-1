# spark_main.py — CS611 A1 (PySpark runner, local[*])
# - Bronze:  CSV -> Parquet (lowercase cols)
# - Silver:  monthly loan_daily(T) = LMS history ≤ T
# - Gold:    features(T) = last-known join (attrs/fins/click ≤ T) + FE (LMS windows & ratios)
#            labels(T)   = overdue in (T, T+MOB] OR leftover at final installment
# Run:
#   python spark_main.py --stage all --start 2023-05-01 --end 2023-11-01
#   python spark_main.py --stage gold --start 2023-05-01 --end 2023-11-01

import os, argparse, re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# -------- Paths (relative to this file's parent as bạn đang để) --------
DATA_DIR   = "/app/data"
BRONZE_DIR = "/app/datamart/bronze"
SILVER_DIR = "/app/datamart/silver"
GOLD_DIR   = "/app/datamart/gold"

RAW_FILES = {
    "lms":         f"{DATA_DIR}/lms_loan_daily.csv",
    "financials":  f"{DATA_DIR}/features_financials.csv",   # auto-fallback handled in bronze
    "attributes":  f"{DATA_DIR}/features_attributes.csv",
    "clickstream": f"{DATA_DIR}/feature_clickstream.csv",
}

# -------- Utils --------
def month_starts(start, end):
    d = datetime.fromisoformat(start).replace(day=1)
    e = datetime.fromisoformat(end)
    out = []
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

def parse_any_date(col):
    # parse m/d/yy -> m/d/yyyy -> generic
    c1 = F.to_timestamp(col, "M/d/yy")
    c2 = F.when(c1.isNull(), F.to_timestamp(col, "M/d/yyyy")).otherwise(c1)
    c3 = F.when(c2.isNull(), F.to_timestamp(col)).otherwise(c2)
    return c3

# strip to numeric safely (keep digits, dot, minus)
@F.udf(DoubleType())
def to_num_safe_udf(x):
    if x is None:
        return None
    s = re.sub(r"[^0-9\.\-]", "", str(x))
    try:
        return float(s) if s != "" else None
    except Exception:
        return None

@F.udf(IntegerType())
def parse_credit_history_to_months_udf(s):
    if s is None:
        return None
    try:
        st = str(s).lower()
        y = re.search(r"(\d+)\s*year", st)
        m = re.search(r"(\d+)\s*month", st)
        years = int(y.group(1)) if y else 0
        months = int(m.group(1)) if m else 0
        return years*12 + months
    except:
        return None

# -------- BRONZE --------
def bronze_ingest(sp):
    os.makedirs(BRONZE_DIR, exist_ok=True)

    # fallback: features_financial.csv
    if not os.path.exists(RAW_FILES["financials"]):
        alt = f"{DATA_DIR}/features_financial.csv"
        if os.path.exists(alt):
            RAW_FILES["financials"] = alt

    for name, path in RAW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] missing raw: {path}")
            continue
        df = sp.read.option("header", True).csv(path)
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip().lower())
        out = f"{BRONZE_DIR}/{name}"
        df.write.mode("overwrite").parquet(out)
        print(f"[BRONZE] {name} -> {out} rows≈{df.count()}")

# -------- SILVER --------
def silver_backfill(sp, start, end):
    os.makedirs(f"{SILVER_DIR}/loan_daily", exist_ok=True)
    lms = sp.read.parquet(f"{BRONZE_DIR}/lms")
    lms = (lms
           .withColumn("post_date", parse_any_date(F.col("snapshot_date")))
           .withColumn("loan_start_date", parse_any_date(F.col("loan_start_date")))
           .dropDuplicates())

    for T in month_starts(start, end):
        Tlit = F.to_timestamp(F.lit(T))
        out_dir = f"{SILVER_DIR}/loan_daily/snapshot_date={T}"
        lms_T = (lms
                 .withColumn("_use_post", F.col("post_date").isNotNull())
                 .filter((F.col("_use_post") & (F.col("post_date") <= Tlit)) |
                         (~F.col("_use_post") & F.col("loan_start_date").isNotNull() &
                          (F.col("loan_start_date") <= Tlit)))
                 .drop("_use_post"))
        lms_T.write.mode("overwrite").parquet(f"{out_dir}/loan_daily.parquet")
        print(f"[SILVER] {T} rows={lms_T.count()} -> {out_dir}")

# -------- GOLD helpers --------
def last_known_on_or_before(df, Tlit):
    # keep last row ≤ T per customer_id using snapshot_date
    if "customer_id" not in df.columns or "snapshot_date" not in df.columns:
        return df.dropDuplicates(["customer_id"]) if "customer_id" in df.columns else df
    df2 = (df
           .withColumn("snapshot_date_ts", parse_any_date(F.col("snapshot_date")))
           .filter(F.col("snapshot_date_ts").isNotNull() & (F.col("snapshot_date_ts") <= Tlit)))
    w = Window.partitionBy("customer_id").orderBy(F.col("snapshot_date_ts").asc())
    return (df2
            .withColumn("_rn", F.row_number().over(w))
            .withColumn("_max", F.max("_rn").over(Window.partitionBy("customer_id")))
            .filter(F.col("_rn") == F.col("_max"))
            .drop("_rn","_max","snapshot_date_ts"))

def make_labels_T(sp, T, mob):
    Tlit = F.to_timestamp(F.lit(T))
    endlit = F.to_timestamp(F.lit((datetime.fromisoformat(T) + relativedelta(months=mob)).strftime("%Y-%m-%d")))
    lms_all = (sp.read.parquet(f"{BRONZE_DIR}/lms")
               .withColumn("post_date", parse_any_date(F.col("snapshot_date")))
               .withColumn("installment_num", F.col("installment_num").cast("int"))
               .withColumn("tenure", F.col("tenure").cast("int"))
               .withColumn("overdue_amt", to_num_safe_udf(F.col("overdue_amt")))
               .withColumn("balance", to_num_safe_udf(F.col("balance"))))

    future = (lms_all
              .filter((F.col("post_date") > Tlit) & (F.col("post_date") <= endlit))
              .withColumn("overdue_flag", (F.col("overdue_amt") > F.lit(0)).cast("int"))
              .groupBy("loan_id").agg(F.max("overdue_flag").alias("overdue_flag")))
    leftover = (lms_all
                .filter(F.col("installment_num") == F.col("tenure"))
                .withColumn("leftover_flag", (F.col("balance") > F.lit(0)).cast("int"))
                .groupBy("loan_id").agg(F.max("leftover_flag").alias("leftover_flag")))
    labels = (future.join(leftover, "loan_id", "outer").na.fill(0)
              .withColumn("label_default",
                          ((F.col("overdue_flag") > 0) | (F.col("leftover_flag") > 0)).cast("int"))
              .select("loan_id","label_default"))
    return labels

# ---- FE: LMS window features (≤ T) ----
def make_lms_window_features_spark(lms_hist_sdf, T):
    T_dt = datetime.fromisoformat(T)
    Tlit = F.to_timestamp(F.lit(T))
    T6 = datetime.fromisoformat(T) - relativedelta(months=6)
    T3 = datetime.fromisoformat(T) - relativedelta(months=3)

    df = (lms_hist_sdf
          .withColumn("post_date", parse_any_date(F.col("post_date")))
          .filter(F.col("post_date").isNotNull() & (F.col("post_date") <= Tlit))
          .withColumn("overdue_amt", to_num_safe_udf(F.col("overdue_amt")))
          .withColumn("due_amt",     to_num_safe_udf(F.col("due_amt")))
          .withColumn("paid_amt",    to_num_safe_udf(F.col("paid_amt")))
          .withColumn("balance",     to_num_safe_udf(F.col("balance")))
          .withColumn("installment_num", F.col("installment_num").cast("int"))
          .withColumn("tenure",          F.col("tenure").cast("int"))
          .withColumn("loan_amt",        to_num_safe_udf(F.col("loan_amt"))))

    # last row ≤ T per loan
    dtcol = F.col("post_date")
    w_last = Window.partitionBy("loan_id").orderBy(dtcol.asc())
    last_row = (df
        .withColumn("_rn", F.row_number().over(w_last))
        .withColumn("_max", F.max("_rn").over(Window.partitionBy("loan_id")))
        .filter(F.col("_rn")==F.col("_max"))
        .select(
            "loan_id",
            F.col("installment_num").alias("inst_num_last"),
            "tenure",
            F.col("balance").alias("balance_last"),
            F.col("overdue_amt").alias("overdue_last"),
            F.col("due_amt").alias("due_last"),
            F.col("paid_amt").alias("paid_last"),
            F.col("loan_amt").alias("loan_amt_last"),
        )
        .withColumn("remaining_tenor",
            F.when(F.col("tenure").isNotNull() & F.col("inst_num_last").isNotNull(),
                   F.greatest(F.col("tenure")-F.col("inst_num_last"), F.lit(0)))
             .otherwise(F.lit(None).cast("double")))
    )

    # 6 months window
    w6 = (df.filter((F.col("post_date") > F.lit(T6)) & (F.col("post_date") <= Tlit))
          .withColumn("ovf", (F.col("overdue_amt") > 0).cast("int"))
          .groupBy("loan_id").agg(
              F.sum("ovf").alias("num_overdue_6m"),
              F.max("overdue_amt").alias("max_overdue_6m"),
              F.sum("due_amt").alias("sum_due_6m"),
              F.sum("paid_amt").alias("sum_paid_6m"),
          )
          .withColumn("paid_to_due_6m",
              F.when(F.col("sum_due_6m")==0, F.lit(None).cast("double"))
               .otherwise(F.col("sum_paid_6m")/F.col("sum_due_6m")))
    )

    # 3 months window
    w3 = (df.filter((F.col("post_date") > F.lit(T3)) & (F.col("post_date") <= Tlit))
          .withColumn("ovf", (F.col("overdue_amt") > 0).cast("int"))
          .groupBy("loan_id").agg(
              F.sum("ovf").alias("num_overdue_3m"),
              F.max("overdue_amt").alias("max_overdue_3m"),
              F.sum("due_amt").alias("sum_due_3m"),
              F.sum("paid_amt").alias("sum_paid_3m"),
          )
          .withColumn("paid_to_due_3m",
              F.when(F.col("sum_due_3m")==0, F.lit(None).cast("double"))
               .otherwise(F.col("sum_paid_3m")/F.col("sum_due_3m")))
    )

    fe = (last_row
          .join(w6, "loan_id", "left")
          .join(w3, "loan_id", "left"))
    return fe

# ---- FE: derived from financials/click + join LMS FE ----
def add_derived_features_spark(feats_sdf, lms_hist_sdf, T):
    out = feats_sdf

    # Normalize obvious financial numeric columns if present
    for c in ["annual_income","monthly_inhand_salary","num_bank_accounts","num_credit_card",
              "interest_rate","num_of_loan","delay_from_due_date","num_of_delayed_payment",
              "changed_credit_limit","num_credit_inquiries","outstanding_debt",
              "credit_utilization_ratio","total_emi_per_month","amount_invested_monthly","monthly_balance"]:
        if c in out.columns:
            out = out.withColumn(c, to_num_safe_udf(F.col(c)))

    # credit history months (if only text available)
    if "credit_history_months" not in out.columns and "credit_history_age" in out.columns:
        out = out.withColumn("credit_history_months", parse_credit_history_to_months_udf(F.col("credit_history_age")))

    # categorical normalizations (kept as string; model will encode later)
    if "credit_mix" in out.columns:
        out = out.withColumn("credit_mix",
              F.when((F.col("credit_mix").isNull()) | (F.col("credit_mix")=="_"), F.lit("unknown"))
               .otherwise(F.col("credit_mix")))
    if "payment_of_min_amount" in out.columns:
        out = out.withColumn("payment_of_min_amount",
              F.when(F.lower(F.col("payment_of_min_amount"))=="nm", F.lit("no"))
               .otherwise(F.col("payment_of_min_amount")))

    # ratios
    if "outstanding_debt" in out.columns and "monthly_inhand_salary" in out.columns:
        out = out.withColumn("dti",
              F.when(F.col("monthly_inhand_salary")==0, F.lit(None).cast("double"))
               .otherwise(F.col("outstanding_debt")/F.col("monthly_inhand_salary")))
    if "total_emi_per_month" in out.columns and "monthly_inhand_salary" in out.columns:
        out = out.withColumn("emi_to_income",
              F.when(F.col("monthly_inhand_salary")==0, F.lit(None).cast("double"))
               .otherwise(F.col("total_emi_per_month")/F.col("monthly_inhand_salary")))

    # type_of_loan -> num_loan_types
    tol_col = [c for c in out.columns if c.lower()=="type_of_loan"]
    if tol_col:
        c = tol_col[0]
        out = out.withColumn("num_loan_types",
              F.when(F.col(c).isNull(), F.lit(0))
               .otherwise(F.size(F.expr(f"filter(transform(split({c}, ','), x -> trim(x)), x -> x <> '')"))))

    # clickstream summary fe_1..fe_20
    fe_cols = [c for c in out.columns if re.fullmatch(r"fe_\d+", c)]
    if fe_cols:
        # sum / mean / std / positive count
        exprs = [F.col(c).cast("double") for c in fe_cols]
        out = (out
               .withColumn("click_sum", F.expr("+".join([f"coalesce(double({c}),0)" for c in fe_cols])))
               .withColumn("click_mean", F.col("click_sum")/F.lit(len(fe_cols)))
               .withColumn("click_posct", sum([(F.col(c).cast("double")>0).cast("int") for c in fe_cols]))
              )
        # simple population-std approximation
        mean_expr = F.col("click_mean")
        out = out.withColumn(
            "click_std",
            F.sqrt(sum([(F.coalesce(F.col(c).cast("double"), F.lit(0.0)) - mean_expr)**2 for c in fe_cols]) / F.lit(len(fe_cols)))
        )

    # join LMS FE (time-safe)
    fe_lms = make_lms_window_features_spark(lms_hist_sdf, T)
    out = out.join(fe_lms, "loan_id", "left")
    return out

# -------- GOLD main --------
def build_features_T(sp, T):
    Tlit = F.to_timestamp(F.lit(T))
    # silver history at T
    lms_hist = sp.read.parquet(f"{SILVER_DIR}/loan_daily/snapshot_date={T}/loan_daily.parquet")

    # base loans: lấy last row ≤ T theo post_date (fallback loan_start_date)
    lms_hist = (lms_hist
                .withColumn("post_date", parse_any_date(F.col("post_date")))
                .withColumn("loan_start_date", parse_any_date(F.col("loan_start_date"))))
    dtcol = F.when(F.col("post_date").isNotNull(), F.col("post_date")).otherwise(F.col("loan_start_date"))
    w_loan = Window.partitionBy("loan_id").orderBy(dtcol.asc())
    base = (lms_hist
            .withColumn("_rn", F.row_number().over(w_loan))
            .withColumn("_max", F.max("_rn").over(Window.partitionBy("loan_id")))
            .filter(F.col("_rn") == F.col("_max"))
            .drop("_rn","_max"))

    # sources last-known ≤ T
    attrs = sp.read.parquet(f"{BRONZE_DIR}/attributes")
    fins  = sp.read.parquet(f"{BRONZE_DIR}/financials")
    click = sp.read.parquet(f"{BRONZE_DIR}/clickstream")

    attrs_T = last_known_on_or_before(attrs, Tlit)
    fins_T  = last_known_on_or_before(fins,  Tlit)
    click_T = last_known_on_or_before(click, Tlit)

    # join by customer_id
    feats = base
    for d in [attrs_T, fins_T, click_T]:
        if "customer_id" in feats.columns and "customer_id" in d.columns:
            feats = feats.join(d.drop("snapshot_date"), on="customer_id", how="left")

    # add derived features (ratios/click summary + LMS window FE)
    feats = add_derived_features_spark(feats, lms_hist, T)
    return feats

def gold_backfill(sp, start, end, mob):
    os.makedirs(f"{GOLD_DIR}/feature_store", exist_ok=True)
    os.makedirs(f"{GOLD_DIR}/label_store", exist_ok=True)

    for T in month_starts(start, end):
        # features
        feats = build_features_T(sp, T).withColumn("snapshot_date", F.to_timestamp(F.lit(T)))
        # labels
        labels = make_labels_T(sp, T, mob).withColumn("snapshot_date", F.to_timestamp(F.lit(T)))

        out_f = f"{GOLD_DIR}/feature_store/snapshot_date={T}"
        out_l = f"{GOLD_DIR}/label_store/snapshot_date={T}"
        feats.write.mode("overwrite").parquet(f"{out_f}/features.parquet")
        labels.write.mode("overwrite").parquet(f"{out_l}/labels.parquet")
        print(f"[GOLD] {T} -> features={feats.count()} labels={labels.count()}")

# -------- CLI --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bronze","silver","gold","all"], default="all")
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end",   type=str, default="2024-12-01")
    ap.add_argument("--mob",   type=int, default=6)
    args = ap.parse_args()

    sp = spark()
    sp.sparkContext.setLogLevel("ERROR")
    try:
        if args.stage in ("bronze","all"):
            bronze_ingest(sp)
        if args.stage in ("silver","all"):
            silver_backfill(sp, args.start, args.end)
        if args.stage in ("gold","all"):
            gold_backfill(sp, args.start, args.end, args.mob)
    finally:
        sp.stop()