import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

monitor_path = Path("datamart/gold/monitor_store/monitoring_history.parquet")
df = pd.read_parquet(monitor_path)
df = df.sort_values("snapshot_date")

# 1. AUC & Accuracy trend
plt.figure(figsize=(8,5))
plt.plot(df["snapshot_date"], df["auc"], marker='o', label="AUC")
plt.plot(df["snapshot_date"], df["accuracy"], marker='o', label="Accuracy")
plt.axhline(y=0.8, linestyle='--', label="AUC Threshold 0.8")
plt.title("Model Performance Over Time")
plt.xlabel("Snapshot Date")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AUC_trend.png", dpi=300)

# 2. PSI trend
plt.figure(figsize=(8,5))
plt.bar(df["snapshot_date"], df["psi_pd_score"])
plt.axhline(y=0.2, linestyle='--', label="PSI Threshold 0.2")
plt.title("Data Drift (PSI) Over Time")
plt.xlabel("Snapshot Date")
plt.ylabel("PSI")
plt.legend()
plt.tight_layout()
plt.savefig("PSI_trend.png", dpi=300)

# 3. Default Rate: Actual vs Predicted
plt.figure(figsize=(8,5))
plt.plot(df["snapshot_date"], df["actual_default_rate"], marker='o', label="Actual")
plt.plot(df["snapshot_date"], df["predicted_default_rate"], marker='o', label="Predicted")
plt.title("Default Rate: Actual vs Predicted")
plt.xlabel("Snapshot Date")
plt.ylabel("Default Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("DefaultRate_trend.png", dpi=300)
