# src/external/manual_adapters.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_optional_manual_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"stock_id": str, "證券代碼": str})

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype={"stock_id": str, "證券代碼": str})

    raise ValueError(f"Unsupported manual file format: {path}")