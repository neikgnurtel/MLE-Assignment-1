import os
import pandas as pd

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df
