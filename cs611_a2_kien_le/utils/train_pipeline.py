import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==== PATHS (bên trong container /app) ====
BASE_DIR = Path(__file__).resolve().parents[1]  # /app
FEATURE_STORE_DIR = BASE_DIR / "datamart" / "gold" / "feature_store"
LABEL_STORE_DIR = BASE_DIR / "datamart" / "gold" / "label_store"
MODEL_STORE_DIR = BASE_DIR / "model_store"

ID_COLS = ["loan_id", "snapshot_date"]
TARGET_COL = "label_default"
LEAKY_COLS = [
    "paid_to_due_3m",
    "paid_to_due_6m",
    "num_overdue_3m",
    "num_overdue_6m",
    "max_overdue_6m",
    "max_overdue_3m",
    "overdue_amt",
    "overdue_last",
    "paid_amt",
    "paid_last",
    "sum_paid_3m",   
    "sum_paid_6m",
    "balance_last",         
    "balance",              
    "delay_from_due_date", 
    "outstanding_debt",    
]

def get_snapshot_dates():
    pattern = str(FEATURE_STORE_DIR / "snapshot_date=*")
    paths = glob.glob(pattern)
    dates = []
    for p in paths:
        name = os.path.basename(p)
        if name.startswith("snapshot_date="):
            dates.append(name.split("=", 1)[1])
    dates = sorted(dates)
    if not dates:
        raise RuntimeError(f"No snapshots found in {FEATURE_STORE_DIR}")
    return dates


def load_snapshot(date_str: str) -> pd.DataFrame:
    feat_path = FEATURE_STORE_DIR / f"snapshot_date={date_str}" / "features.parquet"
    label_path = LABEL_STORE_DIR / f"snapshot_date={date_str}" / "labels.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features for {date_str}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels for {date_str}")

    df_f = pd.read_parquet(feat_path)
    df_l = pd.read_parquet(label_path)

    df = df_f.merge(df_l, on=ID_COLS, how="inner")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def load_all():
    dates = get_snapshot_dates()
    dfs = [load_snapshot(d) for d in dates]
    return pd.concat(dfs, ignore_index=True)


def temporal_split(df: pd.DataFrame):
    last = df["snapshot_date"].max()
    train = df[df["snapshot_date"] < last].copy()
    val = df[df["snapshot_date"] == last].copy()
    if train.empty or val.empty:
        raise RuntimeError("Temporal split failed.")
    return train, val


def prepare_xy(df: pd.DataFrame, feature_cols=None, median_values=None):
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"{TARGET_COL} not in columns.")

    drop_cols = ID_COLS + [TARGET_COL] + LEAKY_COLS

    if feature_cols is None:
        # Tự build từ df
        feat_candidates = [c for c in df.columns if c not in drop_cols]
        X_full = df[feat_candidates]
        num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = num_cols
    else:
        # Dùng đúng feature_cols đã chọn từ train
        num_cols = feature_cols

    X = df[num_cols].copy()
    y = df[TARGET_COL].astype(int)

    if median_values is None:
        # Chỉ chạy khi xử lý X_train
        print("Calculating median values from training data...")
        median_values = X.median(numeric_only=True)
    
    X = X.fillna(median_values)
    
    # Trả về median_values để dùng cho tập val
    return X, y, feature_cols, median_values

    return X, y, feature_cols

def train_and_select(X_train, y_train, X_val, y_val):
    results = []

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    results.append(("logreg", lr, lr_auc))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    results.append(("rf", rf, rf_auc))

    print("Validation AUC:")
    for name, _, auc in results:
        print(f"  {name}: {auc:.4f}")

    best_name, best_model, best_auc = max(results, key=lambda x: x[2])
    print(f"✅ Best model: {best_name} (AUC={best_auc:.4f})")
    return best_name, best_model, best_auc


def save_model(model, feature_cols, model_name, val_auc):
    os.makedirs(MODEL_STORE_DIR, exist_ok=True)
    artefact = MODEL_STORE_DIR / "best_model.pkl"
    meta = MODEL_STORE_DIR / "best_model_meta.json"

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "model_name": model_name,
            "val_auc": float(val_auc),
        },
        artefact,
    )

    import json
    with open(meta, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "val_auc": float(val_auc),
                "artefact": str(artefact),
            },
            f,
            indent=2,
        )

    print(f"💾 Saved model to {artefact}")
    print(f"💾 Saved meta   to {meta}")


def run_training():
    print("=== Load GOLD data ===")
    df = load_all()
    print(df.shape)

    print("=== Temporal split ===")
    train_df, val_df = temporal_split(df)

    print("Train snapshots:", sorted(train_df["snapshot_date"].unique()))
    print("Val snapshot:", sorted(val_df["snapshot_date"].unique()))

    print("=== Prepare X, y ===")
    X_train, y_train, feat_cols, train_medians = prepare_xy(train_df)
    X_val, y_val, _ , _ = prepare_xy(val_df, 
                                     feature_cols=feat_cols, 
                                     median_values=train_medians)

    print("X_train:", X_train.shape, "positives:", int(y_train.sum()))
    print("X_val  :", X_val.shape, "positives:", int(y_val.sum()))

    print("=== Train models ===")
    best_name, best_model, best_auc = train_and_select(X_train, y_train, X_val, y_val)

    print("=== Save best ===")
    save_model(best_model, feat_cols, best_name, best_auc)


if __name__ == "__main__":
    run_training()