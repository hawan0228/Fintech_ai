# src/external/twse_client.py
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import requests


def fetch_twse_stock_day(
    stock_id: str,
    year: int,
    month: int = 12,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Optional TWSE fallback.

    注意：
    - 此函式使用 TWSE exchangeReport/STOCK_DAY 形式查詢。
    - 若 TWSE endpoint 格式或可得期間改變，可能回傳空資料。
    - 本專案主線建議使用 FinMind，TWSE 作為補充。
    """
    date_str = f"{year}{month:02d}01"

    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    params = {
        "response": "json",
        "date": date_str,
        "stockNo": str(stock_id),
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame()

    time.sleep(sleep_seconds)

    if "data" not in payload or "fields" not in payload:
        return pd.DataFrame()

    df = pd.DataFrame(payload["data"], columns=payload["fields"])

    if df.empty:
        return df

    df["stock_id"] = str(stock_id)
    df["query_year"] = int(year)
    df["query_month"] = int(month)

    return df