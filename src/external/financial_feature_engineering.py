from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    EXTERNAL_FINANCIAL_VALUE_UNIT,
    EXTERNAL_PAR_VALUE_PER_SHARE,
)
from src.schema import (
    STOCK_ID_COL,
    YEAR_COL,
)


def clean_stock_id(value) -> str:
    text = str(value).strip()
    text = text.replace(".0", "")

    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)

    return text


def to_datetime_safe(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()

    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def normalize_financial_value(value: float | int | str) -> float:
    """
    將 FinMind 財報 value 正規化成「百萬元」。

    常見情況：
    - 若 EXTERNAL_FINANCIAL_VALUE_UNIT = "thousand_ntd":
      value 單位視為千元，轉百萬元要除以 1000。
    - 若 EXTERNAL_FINANCIAL_VALUE_UNIT = "ntd":
      value 單位視為元，轉百萬元要除以 1,000,000。
    - 若 EXTERNAL_FINANCIAL_VALUE_UNIT = "million_ntd":
      value 已是百萬元，直接使用。
    """
    x = pd.to_numeric(value, errors="coerce")

    if pd.isna(x):
        return np.nan

    x = float(x)

    if EXTERNAL_FINANCIAL_VALUE_UNIT == "thousand_ntd":
        return x / 1000.0

    if EXTERNAL_FINANCIAL_VALUE_UNIT == "ntd":
        return x / 1_000_000.0

    if EXTERNAL_FINANCIAL_VALUE_UNIT == "million_ntd":
        return x

    raise ValueError(
        f"Unknown EXTERNAL_FINANCIAL_VALUE_UNIT: {EXTERNAL_FINANCIAL_VALUE_UNIT}"
    )


def safe_divide(numerator, denominator) -> float:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if abs(float(denominator)) < 1e-12:
        return np.nan

    return float(numerator) / float(denominator)


def match_type(text: str, keywords: Iterable[str]) -> bool:
    normalized = str(text).lower().replace(" ", "")

    for keyword in keywords:
        key = str(keyword).lower().replace(" ", "")

        if key in normalized:
            return True

    return False


def prepare_long_financial_df(
    df: pd.DataFrame,
    value_col: str = "value",
    type_col: str = "type",
) -> pd.DataFrame:
    """
    FinMind 財報資料常見格式：
    stock_id, date, type, value
    """
    if df.empty:
        return pd.DataFrame()

    required_cols = ["stock_id", "date", type_col, value_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return pd.DataFrame()

    out = df.copy()
    out = to_datetime_safe(out, "date")

    out["stock_id"] = out["stock_id"].astype(str).map(clean_stock_id)
    out["year"] = out["date"].dt.year.astype("Int64")
    out[type_col] = out[type_col].astype(str)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    out = out.dropna(subset=["stock_id", "year", value_col]).copy()
    out["year"] = out["year"].astype(int)

    return out


def extract_latest_item_by_keywords(
    df: pd.DataFrame,
    keywords: list[str],
    output_name: str,
    value_col: str = "value",
    type_col: str = "type",
) -> pd.DataFrame:
    """
    從 long-format financial statement 中，以 type 關鍵字抽取年度最後一筆數值。
    輸出單位統一為百萬元。
    """
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "year", output_name])

    matched = df[
        df[type_col].apply(lambda text: match_type(text, keywords))
    ].copy()

    if matched.empty:
        return pd.DataFrame(columns=["stock_id", "year", output_name])

    matched = matched.sort_values(["stock_id", "year", "date"])

    latest = (
        matched.groupby(["stock_id", "year"], as_index=False)
        .last()
    )

    latest[output_name] = latest[value_col].apply(normalize_financial_value)

    return latest[["stock_id", "year", output_name]].copy()


def merge_items(item_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = None

    for item_df in item_dfs:
        if item_df is None or item_df.empty:
            continue

        if merged is None:
            merged = item_df.copy()
        else:
            merged = merged.merge(
                item_df,
                on=["stock_id", "year"],
                how="outer",
            )

    if merged is None:
        return pd.DataFrame(columns=["stock_id", "year"])

    return merged


def extract_income_statement_items(financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    從損益表抽取：
    - revenue
    - operating_income
    - net_income
    """
    df = prepare_long_financial_df(financial_df)

    if df.empty:
        return pd.DataFrame(columns=["stock_id", "year"])

    revenue = extract_latest_item_by_keywords(
        df,
        keywords=[
            "營業收入合計",
            "營業收入",
            "收益",
            "revenue",
            "operating revenue",
        ],
        output_name="revenue_million",
    )

    operating_income = extract_latest_item_by_keywords(
        df,
        keywords=[
            "營業利益",
            "營業淨利",
            "利益（損失）",
            "operating income",
            "operating profit",
        ],
        output_name="operating_income_million",
    )

    net_income = extract_latest_item_by_keywords(
        df,
        keywords=[
            "本期淨利",
            "稅後淨利",
            "淨利",
            "net income",
            "profit",
        ],
        output_name="net_income_million",
    )

    return merge_items([revenue, operating_income, net_income])


def extract_balance_sheet_items(balance_df: pd.DataFrame) -> pd.DataFrame:
    """
    從資產負債表抽取：
    - total_assets
    - total_equity
    - total_liabilities
    - share_capital
    """
    df = prepare_long_financial_df(balance_df)

    if df.empty:
        return pd.DataFrame(columns=["stock_id", "year"])

    total_assets = extract_latest_item_by_keywords(
        df,
        keywords=[
            "資產總計",
            "資產合計",
            "total assets",
        ],
        output_name="total_assets_million",
    )

    total_equity = extract_latest_item_by_keywords(
        df,
        keywords=[
            "權益總計",
            "權益總額",
            "權益合計",
            "權益總和",
            "股東權益總計",
            "股東權益合計",
            "股東權益總額",
            "權益總計歸屬於母公司業主",
            "歸屬於母公司業主之權益合計",
            "歸屬於母公司業主之權益總計",
            "母公司業主權益合計",
            "母公司業主權益總計",
            "權益－歸屬於母公司業主",
            "權益總額歸屬於母公司業主",
            "total equity",
            "stockholders equity",
            "shareholders equity",
            "equity attributable to owners of parent",
        ],
        output_name="total_equity_million",
    )

    total_liabilities = extract_latest_item_by_keywords(
        df,
        keywords=[
            "負債總計",
            "負債總額",
            "負債合計",
            "負債總和",
            "total liabilities",
            "liabilities",
        ],
        output_name="total_liabilities_million",
    )

    share_capital = extract_latest_item_by_keywords(
        df,
        keywords=[
            "股本合計",
            "普通股股本",
            "股本",
            "普通股",
            "share capital",
            "capital stock",
            "ordinary share",
            "common stock",
        ],
        output_name="share_capital_million",
    )

    return merge_items([
        total_assets,
        total_equity,
        total_liabilities,
        share_capital,
    ])


def add_average_balance_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算平均資產、平均權益，用於 ROA / ROE。
    """
    out = df.sort_values(["stock_id", "year"]).copy()

    for col in ["total_assets_million", "total_equity_million"]:
        if col in out.columns:
            out[f"avg_{col}"] = (
                out.groupby("stock_id")[col]
                .transform(lambda s: (s + s.shift(1)) / 2.0)
            )

            # 第一年度沒有前期資料時，退而使用期末值。
            out[f"avg_{col}"] = out[f"avg_{col}"].fillna(out[col])

    return out


def calculate_external_financial_features(
    income_items_df: pd.DataFrame,
    balance_items_df: pd.DataFrame,
    price_return_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    產生與原始專題欄位對應的外部財務特徵：

    - 市值(百萬元)
    - 股價營收比
    - 淨值報酬率─稅後
    - 資產報酬率 ROA
    - 營業利益率 OPM
    - 利潤邊際 NPM
    - 稅後淨利成長率
    """
    if income_items_df is None or income_items_df.empty:
        income_items_df = pd.DataFrame(columns=["stock_id", "year"])

    if balance_items_df is None or balance_items_df.empty:
        balance_items_df = pd.DataFrame(columns=["stock_id", "year"])

    df = income_items_df.merge(
        balance_items_df,
        on=["stock_id", "year"],
        how="outer",
    )

    if price_return_df is not None and not price_return_df.empty:
        price_df = price_return_df.copy()
        price_df["stock_id"] = price_df[STOCK_ID_COL].astype(str).map(clean_stock_id)
        price_df["year"] = price_df[YEAR_COL].astype(int)

        price_cols = [
            "stock_id",
            "year",
            "收盤價(元)_年",
        ]

        price_df = price_df[price_cols].drop_duplicates(["stock_id", "year"])

        df = df.merge(price_df, on=["stock_id", "year"], how="left")

    df = add_average_balance_items(df)

    # 股本百萬元 -> 股本元 -> 股數
    if "share_capital_million" in df.columns:
        share_capital_ntd = df["share_capital_million"] * 1_000_000.0
        shares_outstanding = share_capital_ntd / EXTERNAL_PAR_VALUE_PER_SHARE
    else:
        shares_outstanding = np.nan

    df["市值(百萬元)"] = (
        pd.to_numeric(df.get("收盤價(元)_年"), errors="coerce")
        * shares_outstanding
        / 1_000_000.0
    )

    df["股價營收比"] = [
        safe_divide(market_cap, revenue)
        for market_cap, revenue in zip(
            df["市值(百萬元)"],
            df.get("revenue_million", pd.Series(np.nan, index=df.index)),
        )
    ]

    df["淨值報酬率─稅後"] = [
        safe_divide(net_income, avg_equity) * 100.0
        if not pd.isna(safe_divide(net_income, avg_equity))
        else np.nan
        for net_income, avg_equity in zip(
            df.get("net_income_million", pd.Series(np.nan, index=df.index)),
            df.get("avg_total_equity_million", pd.Series(np.nan, index=df.index)),
        )
    ]

    df["資產報酬率 ROA"] = [
        safe_divide(net_income, avg_assets) * 100.0
        if not pd.isna(safe_divide(net_income, avg_assets))
        else np.nan
        for net_income, avg_assets in zip(
            df.get("net_income_million", pd.Series(np.nan, index=df.index)),
            df.get("avg_total_assets_million", pd.Series(np.nan, index=df.index)),
        )
    ]

    df["營業利益率 OPM"] = [
        safe_divide(operating_income, revenue) * 100.0
        if not pd.isna(safe_divide(operating_income, revenue))
        else np.nan
        for operating_income, revenue in zip(
            df.get("operating_income_million", pd.Series(np.nan, index=df.index)),
            df.get("revenue_million", pd.Series(np.nan, index=df.index)),
        )
    ]

    df["利潤邊際 NPM"] = [
        safe_divide(net_income, revenue) * 100.0
        if not pd.isna(safe_divide(net_income, revenue))
        else np.nan
        for net_income, revenue in zip(
            df.get("net_income_million", pd.Series(np.nan, index=df.index)),
            df.get("revenue_million", pd.Series(np.nan, index=df.index)),
        )
    ]

    df["負債/淨值比"] = [
        safe_divide(total_liabilities, total_equity) * 100.0
        if not pd.isna(safe_divide(total_liabilities, total_equity))
        else np.nan
        for total_liabilities, total_equity in zip(
            df.get("total_liabilities_million", pd.Series(np.nan, index=df.index)),
            df.get("total_equity_million", pd.Series(np.nan, index=df.index)),
        )
    ]

    df = df.sort_values(["stock_id", "year"]).copy()

    if "net_income_million" in df.columns:
        df["稅後淨利成長率"] = (
            df.groupby("stock_id")["net_income_million"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            * 100.0
        )
    else:
        df["稅後淨利成長率"] = np.nan

    out = pd.DataFrame()
    out[STOCK_ID_COL] = df["stock_id"]
    out[YEAR_COL] = df["year"].astype(int)

    feature_cols = [
        "市值(百萬元)",
        "股價營收比",
        "淨值報酬率─稅後",
        "資產報酬率 ROA",
        "營業利益率 OPM",
        "利潤邊際 NPM",
        "負債/淨值比",
        "稅後淨利成長率",
    ]

    for col in feature_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    return out.drop_duplicates([STOCK_ID_COL, YEAR_COL]).reset_index(drop=True)
