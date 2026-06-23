# src/external/yfinance_client.py
from __future__ import annotations

import time

import pandas as pd


def fetch_yfinance_price(
    stock_id: str,
    start_date: str,
    end_date: str,
    sleep_seconds: float = 0.35,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "尚未安裝 yfinance，請先執行：pip install yfinance"
        ) from exc

    ticker = f"{stock_id}.TW"

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    time.sleep(sleep_seconds)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # yfinance 可能回傳 MultiIndex columns，這裡攤平。
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "max",
        "Low": "min",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "Trading_Volume",
    }

    df = df.rename(columns=rename_map)
    df["stock_id"] = str(stock_id)

    keep_cols = [
        "date",
        "stock_id",
        "open",
        "max",
        "min",
        "close",
        "adj_close",
        "Trading_Volume",
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()

    return df