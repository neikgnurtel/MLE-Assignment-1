https://github.com/neikgnurtel/mle-assignment
# MLE Assignment 1 – Medallion Backfill (Pandas)

## Local
pip install -r requirements.txt
python main.py --start 2023-05-01 --end 2023-11-01

## By stage
python main.py --stage bronze
python main.py --stage silver --start 2023-05-01 --end 2023-11-01
python main.py --stage gold   --start 2023-05-01 --end 2023-11-01

## Docker
docker compose build
docker compose up

## Sanity Model
python - <<'PY'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T="2023-09-01"
fe=pd.read_parquet(f"datamart/gold/feature_store/snapshot_date={T}/features.parquet")
lb=pd.read_parquet(f"datamart/gold/label_store/snapshot_date={T}/labels.parquet")
df=fe.merge(lb,on="loan_id",how="inner")
y=df["label_default"].astype(int)
X=df.select_dtypes("number").drop(columns=["label_default"],errors="ignore").fillna(0)

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
mdl=LogisticRegression(max_iter=1000,n_jobs=-1).fit(Xtr,ytr)
auc=roc_auc_score(yte, mdl.predict_proba(Xte)[:,1])
print("Rows:",len(df)," PosRate:",y.mean().round(4)," AUC:",round(auc,4))
PY
