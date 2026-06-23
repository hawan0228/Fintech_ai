# src/external/build_external_dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEARMONTH_COL,
    YEAR_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)


def _to_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()

    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def _clean_stock_id(value) -> str:
    text = str(value).strip()
    text = text.replace(".0", "")
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def build_anchor_prices(
    price_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    建立每年 12 月第一個可得交易日的 anchor price。
    用 year 的 12 月價格與下一年 12 月價格計算 Return。
    """
    if price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    df = _to_datetime(df, "date")

    if "close" not in df.columns:
        raise ValueError("price_df must contain close column.")

    df["stock_id"] = df["stock_id"].astype(str).map(_clean_stock_id)
    df["calendar_year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    dec_df = df[
        (df["month"] == 12)
        & (df["calendar_year"].between(start_year, end_year + 1))
    ].copy()

    if dec_df.empty:
        return pd.DataFrame()

    dec_df = dec_df.sort_values(["stock_id", "calendar_year", "date"])

    anchor = (
        dec_df.groupby(["stock_id", "calendar_year"], as_index=False)
        .first()
    )

    anchor = anchor.rename(
        columns={
            "calendar_year": "anchor_year",
            "close": "anchor_close",
        }
    )

    return anchor[["stock_id", "anchor_year", "date", "anchor_close"]].copy()


def build_returns_from_anchor_prices(
    anchor_df: pd.DataFrame,
    ticker_name_map: dict[str, str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Return(year) = Dec price of year+1 / Dec price of year - 1。
    """
    records = []

    for stock_id, group_df in anchor_df.groupby("stock_id"):
        temp = group_df.set_index("anchor_year").sort_index()

        for year in range(start_year, end_year):
            if year not in temp.index or year + 1 not in temp.index:
                continue

            p1 = float(temp.loc[year, "anchor_close"])
            p2 = float(temp.loc[year + 1, "anchor_close"])

            if not np.isfinite(p1) or not np.isfinite(p2) or p1 <= 0:
                continue

            annual_return = (p2 - p1) / p1 * 100.0

            records.append(
                {
                    STOCK_ID_COL: stock_id,
                    STOCK_NAME_COL: ticker_name_map.get(stock_id, stock_id),
                    YEARMONTH_COL: int(f"{year}12"),
                    YEAR_COL: int(year),
                    TARGET_RETURN: annual_return,
                    "收盤價(元)_年": p1,
                }
            )

    return pd.DataFrame(records)


def extract_year_end_per_pbr(
    per_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    從 FinMind TaiwanStockPER 抽取每年 12 月最後一筆 PBR / PER。
    """
    if per_df.empty:
        return pd.DataFrame()

    df = per_df.copy()
    df = _to_datetime(df, "date")
    df["stock_id"] = df["stock_id"].astype(str).map(_clean_stock_id)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df = df[
        (df["month"] == 12)
        & (df["year"].between(start_year, end_year))
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["stock_id", "year", "date"])
    last_df = df.groupby(["stock_id", "year"], as_index=False).last()

    out = pd.DataFrame()
    out[STOCK_ID_COL] = last_df["stock_id"]
    out[YEAR_COL] = last_df["year"]

    # FinMind 欄位名稱可能是 PBratio / PBratio / PriceBookRatio，這裡做彈性處理。
    pbr_candidates = ["PBratio", "PBratio", "PBR", "PriceBookRatio"]
    pe_candidates = ["PER", "PEratio", "PriceEarningRatio"]

    pbr_col = next((col for col in pbr_candidates if col in last_df.columns), None)
    pe_col = next((col for col in pe_candidates if col in last_df.columns), None)

    out["股價淨值比"] = pd.to_numeric(last_df[pbr_col], errors="coerce") if pbr_col else np.nan
    out["Unknown masked parameter"] = pd.to_numeric(last_df[pe_col], errors="coerce") if pe_col else np.nan

    return out


def extract_revenue_growth(
    revenue_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    從月營收資料建立年度營收成長率 proxy。
    """
    if revenue_df.empty:
        return pd.DataFrame()

    df = revenue_df.copy()

    date_col = "date" if "date" in df.columns else None
    if date_col:
        df = _to_datetime(df, "date")
        df["year"] = df["date"].dt.year
    elif "revenue_year" in df.columns:
        df["year"] = pd.to_numeric(df["revenue_year"], errors="coerce")
    else:
        return pd.DataFrame()

    df["stock_id"] = df["stock_id"].astype(str).map(_clean_stock_id)

    revenue_col_candidates = [
        "revenue",
        "Revenue",
        "當月營收",
        "monthly_revenue",
    ]

    revenue_col = next(
        (col for col in revenue_col_candidates if col in df.columns),
        None,
    )

    if revenue_col is None:
        return pd.DataFrame()

    df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")

    annual = (
        df.groupby(["stock_id", "year"], as_index=False)[revenue_col]
        .sum()
        .rename(columns={revenue_col: "annual_revenue"})
    )

    annual = annual.sort_values(["stock_id", "year"])
    annual["營業利益成長率"] = (
        annual.groupby("stock_id")["annual_revenue"].pct_change() * 100.0
    )

    annual = annual[annual["year"].between(start_year, end_year)].copy()

    out = pd.DataFrame()
    out[STOCK_ID_COL] = annual["stock_id"]
    out[YEAR_COL] = annual["year"].astype(int)
    out["股價營收比"] = np.nan
    out["營業利益成長率"] = annual["營業利益成長率"]

    return out


def extract_financial_statement_features(
    financial_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    從 FinMind financial statements 嘗試抽取財務比率 proxy。

    不同資料集欄位可能是:
    stock_id, date, type, value
    所以採用 type 關鍵字對應。
    """
    if financial_df.empty:
        return pd.DataFrame()

    df = financial_df.copy()
    df = _to_datetime(df, "date")

    if "type" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    df["stock_id"] = df["stock_id"].astype(str).map(_clean_stock_id)
    df["year"] = df["date"].dt.year
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[df["year"].between(start_year, end_year)].copy()

    if df.empty:
        return pd.DataFrame()

    type_map = {
        "淨值報酬率─稅後": ["ROE", "ReturnOnEquity", "權益報酬", "淨值報酬"],
        "資產報酬率 ROA": ["ROA", "ReturnOnAssets", "資產報酬"],
        "營業利益率 OPM": ["OperatingProfitMargin", "營業利益率"],
        "利潤邊際 NPM": ["NetProfitMargin", "淨利率", "利潤邊際"],
        "負債/淨值比": ["DebtEquityRatio", "負債權益", "負債淨值"],
        "流動比率": ["CurrentRatio", "流動比率"],
        "速動比率": ["QuickRatio", "速動比率"],
        "存貨週轉率 (次)": ["InventoryTurnover", "存貨週轉"],
        "應收帳款週轉次": ["ReceivableTurnover", "應收帳款週轉"],
        "稅後淨利成長率": ["NetIncomeGrowth", "稅後淨利成長"],
    }

    records = []

    for (stock_id, year), group_df in df.groupby(["stock_id", "year"]):
        record = {
            STOCK_ID_COL: stock_id,
            YEAR_COL: int(year),
        }

        for feature_name, keywords in type_map.items():
            matched = group_df[
                group_df["type"].astype(str).apply(
                    lambda text: any(keyword.lower() in text.lower() for keyword in keywords)
                )
            ]

            if matched.empty:
                record[feature_name] = np.nan
            else:
                record[feature_name] = float(matched.sort_values("date")["value"].iloc[-1])

        records.append(record)

    return pd.DataFrame(records)


def merge_external_features(
    returns_df: pd.DataFrame,
    per_pbr_features: pd.DataFrame,
    revenue_features: pd.DataFrame,
    financial_features: pd.DataFrame,
    engineered_financial_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = returns_df.copy()

    feature_dfs = [
        per_pbr_features,
        revenue_features,
        financial_features,
        engineered_financial_features,
    ]

    for feature_df in feature_dfs:
        if feature_df is not None and not feature_df.empty:
            df = df.merge(
                feature_df,
                on=[STOCK_ID_COL, YEAR_COL],
                how="left",
                suffixes=("", "_new"),
            )

            # 若同一個欄位已有舊值與新值，優先使用新值補舊值缺失。
            new_cols = [col for col in df.columns if col.endswith("_new")]

            for new_col in new_cols:
                base_col = new_col.replace("_new", "")

                if base_col in df.columns:
                    df[base_col] = df[base_col].combine_first(df[new_col])
                    df = df.drop(columns=[new_col])
                else:
                    df = df.rename(columns={new_col: base_col})

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df[TARGET_RETURN] = pd.to_numeric(df[TARGET_RETURN], errors="coerce")

    yearly_mean = df.groupby(YEAR_COL)[TARGET_RETURN].transform("mean")
    df[TARGET_CLASS] = np.where(df[TARGET_RETURN] > yearly_mean, 1, -1)

    df = df.dropna(subset=[TARGET_RETURN, TARGET_CLASS]).copy()

    df[STOCK_ID_COL] = df[STOCK_ID_COL].astype(str).map(_clean_stock_id)
    df[STOCK_NAME_COL] = df[STOCK_NAME_COL].astype(str)

    output_cols = [
        STOCK_ID_COL,
        STOCK_NAME_COL,
        YEARMONTH_COL,
        YEAR_COL,
        *FEATURE_COLUMNS,
        TARGET_RETURN,
        TARGET_CLASS,
    ]

    df = df[output_cols].copy()
    df = df.sort_values([YEAR_COL, STOCK_ID_COL]).reset_index(drop=True)

    return df

def generate_external_profile(df: pd.DataFrame) -> str:
    lines = []

    lines.append("========== External Crawled Dataset Profile ==========")
    lines.append("")
    lines.append(f"Shape: {df.shape}")
    lines.append(f"Years: {df[YEAR_COL].min()} - {df[YEAR_COL].max()}")
    lines.append("")
    lines.append("Rows by year:")
    lines.append(df.groupby(YEAR_COL).size().to_string())
    lines.append("")
    lines.append("Label distribution:")
    lines.append(df[TARGET_CLASS].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Return summary:")
    lines.append(df[TARGET_RETURN].describe().to_string())
    lines.append("")
    lines.append("Feature missing ratio:")
    missing_ratio = df[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    lines.append(missing_ratio.to_string())
    lines.append("")
    lines.append("========== End Profile ==========")

    return "\n".join(lines)