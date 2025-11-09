import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ==== Paths & constants ====

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTION_STORE_DIR = BASE_DIR / "datamart" / "gold" / "prediction_store"
LABEL_STORE_DIR = BASE_DIR / "datamart" / "gold" / "label_store"
MONITOR_STORE_DIR = BASE_DIR / "datamart" / "gold" / "monitor_store"

ID_COLS = ["loan_id", "snapshot_date"]
TARGET_COL = "label_default"


# ==== Helpers ====

def _list_prediction_snapshots():
    pattern = str(PREDICTION_STORE_DIR / "snapshot_date=*")
    paths = glob.glob(pattern)
    dates = []
    for p in paths:
        name = os.path.basename(p)
        if name.startswith("snapshot_date="):
            dates.append(name.split("=", 1)[1])
    return sorted(dates)


def _load_predictions(snapshot_date: str) -> pd.DataFrame:
    path = PREDICTION_STORE_DIR / f"snapshot_date={snapshot_date}" / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found for {snapshot_date}: {path}")
    return pd.read_parquet(path)


def _load_labels(snapshot_date: str) -> pd.DataFrame:
    path = LABEL_STORE_DIR / f"snapshot_date={snapshot_date}" / "labels.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Labels not found for {snapshot_date}: {path}")
    return pd.read_parquet(path)


# ==== Metrics ====

def compute_auc_for_snapshot(snapshot_date: str):
    """
    Join predictions + labels for a snapshot and compute AUC.
    Only works if label_default is already defined for that snapshot.
    """
    print(f"=== AUC check for {snapshot_date} ===")
    df_pred = _load_predictions(snapshot_date)
    df_lab = _load_labels(snapshot_date)

    df = df_pred.merge(df_lab, on=ID_COLS, how="inner")
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"{TARGET_COL} not found after join.")

    y_true = df[TARGET_COL].astype(int)
    y_score = df["pd_score"].astype(float)

    if y_true.nunique() < 2:
        print("Not enough label variation to compute AUC.")
        return None

    auc = roc_auc_score(y_true, y_score)
    print(f"AUC[{snapshot_date}] = {auc:.4f}")
    return auc


def psi(actual, expected, buckets=10):
    """
    Simple PSI for 1D numeric arrays.
    """
    actual = np.array(actual)
    expected = np.array(expected)

    # remove NaN
    actual = actual[~np.isnan(actual)]
    expected = expected[~np.isnan(expected)]

    if len(actual) == 0 or len(expected) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.quantile(expected, quantiles)

    # ensure unique cuts
    cuts = np.unique(cuts)
    if len(cuts) <= 2:
        return 0.0

    actual_counts, _ = np.histogram(actual, bins=cuts)
    expected_counts, _ = np.histogram(expected, bins=cuts)

    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)

    # avoid log(0)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def compute_pd_score_psi():
    """
    Compute PSI on pd_score between two latest prediction snapshots (if >=2 exist).
    """
    snaps = _list_prediction_snapshots()
    if len(snaps) < 2:
        print("Not enough prediction snapshots for PSI (need >=2). Skipping.")
        return None, None, None

    ref_date = snaps[-2]
    cur_date = snaps[-1]

    print(f"=== PSI on pd_score: {ref_date} -> {cur_date} ===")

    df_ref = _load_predictions(ref_date)
    df_cur = _load_predictions(cur_date)

    psi_val = psi(
        actual=df_cur["pd_score"].values,
        expected=df_ref["pd_score"].values,
        buckets=10,
    )
    print(f"PSI(pd_score) {ref_date} -> {cur_date} = {psi_val:.4f}")
    return ref_date, cur_date, psi_val


def save_monitoring_result(auc_result, psi_result):
    """
    Write a short summary table for the latest run to monitoring_summary.parquet.
    """
    os.makedirs(MONITOR_STORE_DIR, exist_ok=True)
    out_path = MONITOR_STORE_DIR / "monitoring_summary.parquet"

    rows = []

    if auc_result is not None:
        snap, auc = auc_result
        rows.append(
            {
                "metric": "AUC",
                "snapshot_date": snap,
                "value": auc,
                "rule": "Alert if AUC drops > 0.05 from training/val baseline",
            }
        )

    if psi_result is not None:
        ref, cur, v = psi_result
        rows.append(
            {
                "metric": "PSI_pd_score",
                "snapshot_date": cur,
                "value": v,
                "reference_snapshot": ref,
                "rule": "Alert if PSI > 0.2 (shift), > 0.3 (severe)",
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        print(f"💾 Saved monitoring summary to {out_path}")
    else:
        print("No monitoring rows to save.")


# ==== Full history builder ====

def build_monitoring_history(threshold: float = 0.5, psi_mode: str = "baseline"):
    """
    Build full monitoring history across all snapshots.

    Creates datamart/gold/monitor_store/monitoring_history.parquet with:
    - snapshot_date
    - auc
    - accuracy (thresholded)
    - actual_default_rate
    - predicted_default_rate
    - psi_pd_score (vs baseline or previous snapshot)
    - psi_reference_snapshot
    """
    os.makedirs(MONITOR_STORE_DIR, exist_ok=True)
    snaps = _list_prediction_snapshots()
    if not snaps:
        print("No prediction snapshots found. Nothing to build history.")
        return

    if psi_mode not in ("baseline", "previous"):
        raise ValueError("psi_mode must be 'baseline' or 'previous'")

    rows = []

    baseline_scores = None
    baseline_snap = None
    prev_scores = None
    prev_snap = None

    for snap in snaps:
        try:
            df_pred = _load_predictions(snap)
            df_lab = _load_labels(snap)
        except FileNotFoundError:
            print(f"[{snap}] Missing predictions or labels, skip.")
            continue

        df = df_pred.merge(df_lab, on=ID_COLS, how="inner")
        if TARGET_COL not in df.columns:
            print(f"[{snap}] {TARGET_COL} not found after join, skip.")
            continue

        y_true = df[TARGET_COL].astype(int)
        y_score = df["pd_score"].astype(float)

        # AUC
        if y_true.nunique() < 2:
            print(f"[{snap}] Not enough label variation for AUC, set NaN.")
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_score)

        # thresholded metrics
        y_pred = (y_score >= threshold).astype(int)
        accuracy = float((y_pred == y_true).mean())
        actual_rate = float(y_true.mean())
        predicted_rate = float(y_pred.mean())

        # PSI
        psi_val = np.nan
        psi_ref_snap = None

        if psi_mode == "baseline":
            if baseline_scores is None:
                baseline_scores = y_score.values
                baseline_snap = snap
            else:
                psi_val = psi(
                    actual=y_score.values,
                    expected=baseline_scores,
                    buckets=10,
                )
                psi_ref_snap = baseline_snap

        elif psi_mode == "previous":
            if prev_scores is not None:
                psi_val = psi(
                    actual=y_score.values,
                    expected=prev_scores,
                    buckets=10,
                )
                psi_ref_snap = prev_snap

        rows.append(
            {
                "snapshot_date": snap,
                "auc": float(auc) if not np.isnan(auc) else np.nan,
                "accuracy": accuracy,
                "actual_default_rate": actual_rate,
                "predicted_default_rate": predicted_rate,
                "psi_pd_score": float(psi_val) if not np.isnan(psi_val) else np.nan,
                "psi_reference_snapshot": psi_ref_snap,
            }
        )

        prev_scores = y_score.values
        prev_snap = snap

    if not rows:
        print("No valid monitoring rows to save.")
        return

    hist_df = pd.DataFrame(rows)
    out_path = MONITOR_STORE_DIR / "monitoring_history.parquet"
    hist_df.to_parquet(out_path, index=False)
    print(f"💾 Saved monitoring history to {out_path}")


# ==== Entry point for Airflow ====

def run_monitoring():
    """
    Entry point cho Airflow task run_monitoring.
    - Tính AUC snapshot mới nhất (nếu có label)
    - Tính PSI giữa 2 snapshot gần nhất
    - Lưu summary
    - Build full monitoring history (best-effort)
    """
    print("=== Start monitoring ===")

    snaps = _list_prediction_snapshots()
    if not snaps:
        print("No prediction snapshots found. Nothing to monitor.")
        return

    latest = snaps[-1]

    # 1) AUC latest snapshot
    try:
        auc = compute_auc_for_snapshot(latest)
        auc_result = (latest, auc) if auc is not None else None
    except FileNotFoundError:
        print(f"No labels found for {latest}, skip AUC.")
        auc_result = None

    # 2) PSI last 2 snapshots
    psi_result = compute_pd_score_psi()

    # 3) Save short summary
    save_monitoring_result(auc_result, psi_result)

    # 4) Build full history (không fail DAG nếu lỗi)
    try:
        build_monitoring_history(threshold=0.5, psi_mode="baseline")
    except Exception as e:
        print(f"Warning: failed to build monitoring history: {e}")


if __name__ == "__main__":
    run_monitoring()