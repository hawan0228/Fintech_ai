# src/external/external_config.py
from __future__ import annotations

from pathlib import Path

from src.config import (
    EXTERNAL_RAW_DIR,
    EXTERNAL_PROCESSED_DIR,
    EXTERNAL_TICKERS_PATH,
)

# 建議先抓較新的資料，因為 TWSE 官方歷史個股日資料頁面可得期間可能有限，
# FinMind 通常較適合作為主要 API 資料源。
DEFAULT_EXTERNAL_START_YEAR = 2018
DEFAULT_EXTERNAL_END_YEAR = 2024

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

EXTERNAL_RAW_PRICE_DIR = EXTERNAL_RAW_DIR / "prices"
EXTERNAL_RAW_PER_DIR = EXTERNAL_RAW_DIR / "per_pbr"
EXTERNAL_RAW_FINANCIAL_DIR = EXTERNAL_RAW_DIR / "financials"
EXTERNAL_RAW_REVENUE_DIR = EXTERNAL_RAW_DIR / "revenue"

for directory in [
    EXTERNAL_RAW_DIR,
    EXTERNAL_PROCESSED_DIR,
    EXTERNAL_RAW_PRICE_DIR,
    EXTERNAL_RAW_PER_DIR,
    EXTERNAL_RAW_FINANCIAL_DIR,
    EXTERNAL_RAW_REVENUE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)