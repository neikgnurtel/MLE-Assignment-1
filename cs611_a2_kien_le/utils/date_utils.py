from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

def month_starts(start_str: str, end_str: str):
    d = datetime.fromisoformat(start_str).replace(day=1)
    end = datetime.fromisoformat(end_str)
    out = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d = (d + relativedelta(months=1)).replace(day=1)
    return out

def month_floor(x):
    """Trả về đầu tháng cho Series lẫn scalar"""
    x = pd.to_datetime(x)
    if isinstance(x, pd.Series):
        return x.dt.to_period('M').dt.to_timestamp()
    else:
        return pd.Timestamp(x).to_period('M').to_timestamp()

def parse_any_date(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", format="%m/%d/%y")
    mask = s.isna() & series.notna()
    if mask.any():
        s2 = pd.to_datetime(series[mask], errors="coerce", format="%m/%d/%Y")
        s.loc[mask] = s2
    mask2 = s.isna() & series.notna()
    if mask2.any():
        s3 = pd.to_datetime(series[mask2], errors="coerce")
        s.loc[mask2] = s3
    return s