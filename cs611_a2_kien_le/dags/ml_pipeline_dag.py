from __future__ import annotations
from datetime import datetime, timedelta
import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---- Ensure project root on sys.path ----
DAGS_DIR = Path(__file__).resolve().parent      # /opt/airflow/dags
PROJECT_ROOT = DAGS_DIR.parent                  # /opt/airflow
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---- Wrappers (import at runtime, not parse time) ----

# === ETL ===
def etl_bronze_wrapper(**_):
    from main import bronze_ingest
    print(">>> Running Bronze Ingest...")
    bronze_ingest()

def etl_silver_wrapper(**_):
    from main import silver_backfill
    print(">>> Running Silver Backfill...")
    silver_backfill()

def etl_gold_wrapper(**_):
    from main import gold_backfill
    print(">>> Running Gold Backfill...")
    gold_backfill()

# === ML ===
def training_wrapper(**_):
    from utils.train_pipeline import run_training
    print(">>> Running Model Training...")
    run_training()

def inference_wrapper(**_):
    from utils.inference_pipeline import run_inference
    print(">>> Running Inference...")
    run_inference()

def monitoring_wrapper(**_):
    from utils.monitor_pipeline import run_monitoring
    print(">>> Running Monitoring...")
    run_monitoring()


# ---- DAG Definition ----

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline_dag",
    description="CS611 A2: End-to-End ETL -> Train -> Inference -> Monitoring",
    default_args=default_args,
    schedule_interval="@monthly",
    start_date=datetime(2023, 11, 1),
    catchup=True,
    tags=["cs611", "mle", "assignment2", "e2e"],
) as dag:

    etl_bronze_task = PythonOperator(
        task_id="etl_bronze",
        python_callable=etl_bronze_wrapper,
    )

    etl_silver_task = PythonOperator(
        task_id="etl_silver",
        python_callable=etl_silver_wrapper,
    )

    etl_gold_task = PythonOperator(
        task_id="etl_gold",
        python_callable=etl_gold_wrapper,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=training_wrapper,
    )

    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=inference_wrapper,
    )

    monitor_task = PythonOperator(
        task_id="run_monitoring",
        python_callable=monitoring_wrapper,
    )

    # Full dependency chain
    etl_bronze_task >> etl_silver_task >> etl_gold_task >> train_task >> inference_task >> monitor_task