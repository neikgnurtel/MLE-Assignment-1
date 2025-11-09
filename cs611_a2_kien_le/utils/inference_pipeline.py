import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_STORE_DIR = BASE_DIR / "datamart" / "gold" / "feature_store"
PREDICTION_STORE_DIR = BASE_DIR / "datamart" / "gold" / "prediction_store"
MODEL_STORE_DIR = BASE_DIR / "model_store"

ID_COLS = ["loan_id", "snapshot_date"]


def get_latest_snapshot():
    pattern = str(FEATURE_STORE_DIR / "snapshot_date=*")
    paths = glob.glob(pattern)
    if not paths:
        raise RuntimeError("No feature_store snapshots found.")

    dates = []
    for p in paths:
        name = os.path.basename(p)
        if name.startswith("snapshot_date="):
            dates.append(name.split("=", 1)[1])

    if not dates:
        raise RuntimeError("No valid snapshot_date folders found in feature_store.")

    latest = sorted(dates)[-1]
    print(f"Using latest snapshot for inference: {latest}")
    return latest


def load_features(snapshot_date: str) -> pd.DataFrame:
    feat_path = FEATURE_STORE_DIR / f"snapshot_date={snapshot_date}" / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found for {snapshot_date}: {feat_path}")
    df = pd.read_parquet(feat_path)
    return df


def load_model():
    artefact = MODEL_STORE_DIR / "best_model.pkl"
    if not artefact.exists():
        raise FileNotFoundError(f"Model artefact not found: {artefact}")

    bundle = joblib.load(artefact)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_name = bundle.get("model_name", "unknown")
    val_auc = bundle.get("val_auc", None)

    print(f"Loaded model: {model_name}, val_auc={val_auc}")
    return model, feature_cols


def prepare_inference_matrix(df_feat: pd.DataFrame, feature_cols):
    # ensure snapshot_date is datetime for consistency
    if "snapshot_date" in df_feat.columns:
        df_feat["snapshot_date"] = pd.to_datetime(df_feat["snapshot_date"])

    # keep only columns the model was trained on
    X = df_feat.reindex(columns=feature_cols, fill_value=np.nan)

    # impute NaN the same simple way as training (median)
    X = X.fillna(X.median(numeric_only=True))

    return X


def save_predictions(df_pred: pd.DataFrame, snapshot_date: str):
    out_dir = PREDICTION_STORE_DIR / f"snapshot_date={snapshot_date}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    df_pred.to_parquet(out_path, index=False)
    print(f"💾 Saved predictions to {out_path}")


def run_inference():
    print("=== Inference: load model ===")
    model, feature_cols = load_model()

    print("=== Detect latest snapshot ===")
    snapshot_date = get_latest_snapshot()

    print("=== Load features ===")
    df_feat = load_features(snapshot_date)

    print("=== Prepare matrix X ===")
    X = prepare_inference_matrix(df_feat, feature_cols)

    print("=== Predict probabilities ===")
    scores = model.predict_proba(X)[:, 1]

    # attach IDs + score
    if not set(ID_COLS).issubset(df_feat.columns):
        raise RuntimeError(f"Missing ID columns in features: {ID_COLS}")

    df_pred = df_feat[ID_COLS].copy()
    df_pred["pd_score"] = scores  # probability of default

    print(df_pred.head())

    print("=== Save prediction snapshot ===")
    save_predictions(df_pred, snapshot_date)


if __name__ == "__main__":
    run_inference()