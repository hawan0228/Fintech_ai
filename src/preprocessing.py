from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DROP_YEARMONTH
from src.schema import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS_BEFORE_YEAR,
    REQUIRED_COLUMNS_AFTER_CLEANING,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    TARGET_CLASS,
    TARGET_RETURN,
    YEARMONTH_COL,
    YEAR_COL,
)


# =========================
# Column name utilities
# =========================

def normalize_column_name(col: object) -> str:
    """
    標準化欄位名稱：
    1. 轉成字串
    2. 移除前後空白
    3. 把換行、tab、多個空白壓成單一空白
    4. 移除部分不可見字元
    """
    col = str(col)
    col = col.replace("\u3000", " ")
    col = col.replace("\xa0", " ")
    col = re.sub(r"[\r\n\t]+", " ", col)
    col = re.sub(r"\s+", " ", col)
    return col.strip()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    將原始 Excel 欄位名稱統一轉成內部標準欄位名稱。
    這裡已針對你提供的實際欄位做 alias 對應。
    """
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    alias_map = {
        # ID columns
        "證券代碼 ": STOCK_ID_COL,
        "公司簡稱": STOCK_NAME_COL,
        "名稱": STOCK_NAME_COL,
        "年月 ": YEARMONTH_COL,

        # Unknown column
        "Unknown masked parameter": "Unknown masked parameter",
        "Unknown masked Parameter": "Unknown masked parameter",
        "Unknown Masked Parameter": "Unknown masked parameter",

        # Original feature aliases from your data
        "M淨值報酬率─稅後": "淨值報酬率─稅後",
        "M淨值報酬率-稅後": "淨值報酬率─稅後",
        "淨值報酬率-稅後": "淨值報酬率─稅後",
        "淨值報酬率－稅後": "淨值報酬率─稅後",

        "資產報酬率ROA": "資產報酬率 ROA",
        "資產報酬率 ROA": "資產報酬率 ROA",

        "營業利益率OPM": "營業利益率 OPM",
        "營業利益率 OPM": "營業利益率 OPM",

        "利潤邊際NPM": "利潤邊際 NPM",
        "利潤邊際 NPM": "利潤邊際 NPM",

        "M流動比率": "流動比率",
        "M速動比率": "速動比率",

        "M存貨週轉率 (次)": "存貨週轉率 (次)",
        "M存貨週轉率(次)": "存貨週轉率 (次)",
        "存貨週轉率(次)": "存貨週轉率 (次)",

        "M應收帳款週轉次": "應收帳款週轉次",
        "應收帳款週轉率": "應收帳款週轉次",

        "M營業利益成長率": "營業利益成長率",
        "M稅後淨利成長率": "稅後淨利成長率",

        # Target aliases
        "return": TARGET_RETURN,
        "Return ": TARGET_RETURN,
        "ReturnMean_year_label": TARGET_CLASS,
        "returnMean_year_Label": TARGET_CLASS,
        "ReturnMean_year_Label ": TARGET_CLASS,
    }

    df = df.rename(columns={col: alias_map.get(col, col) for col in df.columns})
    return df


# =========================
# Validation utilities
# =========================

def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    stage_name: str = "data",
) -> None:
    """
    檢查必要欄位是否存在。
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        available_cols = "\n".join([f"- {col}" for col in df.columns])
        missing_cols = "\n".join([f"- {col}" for col in missing])

        raise ValueError(
            f"[{stage_name}] 缺少必要欄位：\n"
            f"{missing_cols}\n\n"
            f"目前資料中的欄位為：\n"
            f"{available_cols}\n\n"
            f"請確認 Excel 欄位名稱是否與 schema.py 或 alias_map 一致。"
        )


def check_label_values(df: pd.DataFrame) -> None:
    """
    檢查 ReturnMean_year_Label 是否只包含 1 與 -1。
    """
    labels = sorted(df[TARGET_CLASS].dropna().unique().tolist())
    allowed = {-1, 1}

    if not set(labels).issubset(allowed):
        raise ValueError(
            f"{TARGET_CLASS} 應只包含 -1 與 1，但目前偵測到：{labels}"
        )


def check_return_values(df: pd.DataFrame) -> None:
    """
    檢查 Return 是否合理。
    專題說明 Return 是百分比數值，Return = -100 代表股價歸零或下市等特殊情況。
    這裡不直接刪除異常值，只提出警告。
    """
    invalid = df[df[TARGET_RETURN] < -100]

    if len(invalid) > 0:
        print(
            f"[Warning] 發現 {len(invalid)} 筆 Return < -100 的資料，"
            f"請檢查是否為資料錯誤。"
        )


# =========================
# Cleaning functions
# =========================

def clean_yearmonth(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 年月 欄位轉成 int，例如 199712。
    """
    df = df.copy()

    def _clean_yearmonth(value: object) -> int | float:
        if pd.isna(value):
            return np.nan

        text = str(value).strip()

        if re.fullmatch(r"\d+\.0", text):
            text = text.split(".")[0]

        digits = re.sub(r"\D", "", text)

        if len(digits) < 6:
            return np.nan

        return int(digits[:6])

    df[YEARMONTH_COL] = df[YEARMONTH_COL].apply(_clean_yearmonth)
    df[YEARMONTH_COL] = pd.to_numeric(df[YEARMONTH_COL], errors="coerce").astype("Int64")
    return df


def create_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    從 年月 建立 year 欄位。
    """
    df = df.copy()
    df[YEAR_COL] = (df[YEARMONTH_COL] // 100).astype("Int64")
    return df


def clean_stock_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    修正股票代碼格式。

    功能：
    1. 2330.0 -> 2330
    2. 去除空白
    3. 若發現 19972330 這種 year + ticker 型態，根據該列 year 修正成 2330。
    """
    df = df.copy()
    fixed_count = 0

    def _clean_one_row(row: pd.Series) -> str:
        nonlocal fixed_count

        value = row[STOCK_ID_COL]

        if pd.isna(value):
            return ""

        text = str(value).strip()

        if re.fullmatch(r"\d+\.0", text):
            text = text.split(".")[0]

        # 移除空白
        text = text.replace(" ", "")

        # 若證券代碼是 19972330，且 year 是 1997，則修正成 2330
        year_value = row.get(YEAR_COL, pd.NA)

        if pd.notna(year_value):
            year_text = str(int(year_value))

            if text.isdigit() and text.startswith(year_text) and len(text) == len(year_text) + 4:
                text = text[len(year_text):]
                fixed_count += 1

        return text

    df[STOCK_ID_COL] = df.apply(_clean_one_row, axis=1)

    if fixed_count > 0:
        print(f"[Info] 已修正疑似 year+ticker 格式的證券代碼：{fixed_count} 筆")

    return df


def clean_stock_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理公司簡稱。
    """
    df = df.copy()
    df[STOCK_NAME_COL] = (
        df[STOCK_NAME_COL]
        .astype(str)
        .str.replace("\u3000", " ", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    return df


def remove_200912(df: pd.DataFrame) -> pd.DataFrame:
    """
    依專題要求刪除 年月 = 200912 的資料。
    """
    df = df.copy()

    before = len(df)
    df = df[df[YEARMONTH_COL] != DROP_YEARMONTH].copy()
    after = len(df)

    print(f"[Info] 已刪除 年月 = {DROP_YEARMONTH} 的資料：{before - after} 筆")
    return df


def convert_percent_or_numeric_series(series: pd.Series) -> pd.Series:
    """
    將可能含有逗號、百分比符號、空白的欄位轉成 numeric。

    範例：
    "1,234.5" -> 1234.5
    "10%" -> 10
    "--" -> NaN
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("％", "", regex=False)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "--": np.nan,
            "-": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
        }
    )

    return pd.to_numeric(cleaned, errors="coerce")


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 16 個財務特徵與 Return 轉成數值。
    """
    df = df.copy()

    numeric_columns = [*FEATURE_COLUMNS, TARGET_RETURN]

    for col in numeric_columns:
        df[col] = convert_percent_or_numeric_series(df[col])

    return df


def convert_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 ReturnMean_year_Label 轉成 int。
    """
    df = df.copy()
    df[TARGET_CLASS] = convert_percent_or_numeric_series(df[TARGET_CLASS])
    df[TARGET_CLASS] = df[TARGET_CLASS].astype("Int64")
    return df


def drop_invalid_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除 Return 或 Label 缺失的資料。
    """
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=[TARGET_RETURN, TARGET_CLASS])
    after = len(df)

    print(f"[Info] 已刪除 Return 或 Label 缺失資料：{before - after} 筆")
    return df


def sort_cleaned_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    依 year、證券代碼排序。
    """
    df = df.copy()
    df = df.sort_values([YEAR_COL, STOCK_ID_COL]).reset_index(drop=True)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    完整 Step 1 清理流程。
    """
    df = df.copy()

    # 1. 欄位名稱標準化
    df = standardize_column_names(df)

    # 2. 檢查欄位
    check_required_columns(
        df,
        required_columns=REQUIRED_COLUMNS_BEFORE_YEAR,
        stage_name="Before cleaning",
    )

    # 3. 年月與 year
    df = clean_yearmonth(df)
    df = create_year_column(df)

    # 4. 清理 ID 欄位
    df = clean_stock_id(df)
    df = clean_stock_name(df)

    # 5. 刪除 200912
    df = remove_200912(df)

    # 6. 數值欄位轉換
    df = convert_numeric_columns(df)
    df = convert_label_column(df)

    # 7. 移除 target 缺失
    df = drop_invalid_target_rows(df)

    # 8. 檢查 label 與 return
    check_label_values(df)
    check_return_values(df)

    # 9. 檢查清理後欄位
    check_required_columns(
        df,
        required_columns=REQUIRED_COLUMNS_AFTER_CLEANING,
        stage_name="After cleaning",
    )

    # 10. 排序
    df = sort_cleaned_data(df)

    return df


# =========================
# Data profile
# =========================

def generate_data_profile(df: pd.DataFrame) -> str:
    """
    產生資料摘要文字。
    """
    lines = []

    lines.append("========== Step 1 Data Profile ==========")
    lines.append("")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Total columns: {df.shape[1]}")
    lines.append("")

    lines.append("Year range:")
    lines.append(f"- Min year: {df[YEAR_COL].min()}")
    lines.append(f"- Max year: {df[YEAR_COL].max()}")
    lines.append("")

    lines.append("Rows by year:")
    year_counts = df[YEAR_COL].value_counts().sort_index()
    for year, count in year_counts.items():
        lines.append(f"- {year}: {count} rows")
    lines.append("")

    lines.append("Label distribution:")
    label_counts = df[TARGET_CLASS].value_counts().sort_index()
    for label, count in label_counts.items():
        lines.append(f"- label {label}: {count} rows")
    lines.append("")

    lines.append("Return summary:")
    return_summary = df[TARGET_RETURN].describe()
    lines.append(str(return_summary))
    lines.append("")

    lines.append("Missing values in feature columns:")
    missing = df[FEATURE_COLUMNS].isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        lines.append("- No missing values in feature columns.")
    else:
        for col, count in missing.items():
            lines.append(f"- {col}: {count}")

    lines.append("")
    lines.append("Feature columns after standardization:")
    for col in FEATURE_COLUMNS:
        lines.append(f"- {col}")

    lines.append("")
    lines.append("First 5 rows after cleaning:")
    preview_cols = [
        STOCK_ID_COL,
        STOCK_NAME_COL,
        YEARMONTH_COL,
        YEAR_COL,
        TARGET_RETURN,
        TARGET_CLASS,
    ]
    lines.append(df[preview_cols].head().to_string(index=False))

    lines.append("")
    lines.append("========== End of Profile ==========")

    return "\n".join(lines)


def save_data_profile(profile_text: str, output_path: str | Path) -> None:
    """
    儲存資料摘要。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(profile_text)