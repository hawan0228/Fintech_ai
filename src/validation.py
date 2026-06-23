from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.schema import (
    YEAR_COL,
    STOCK_ID_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)


ValidationMode = Literal["next_year", "remaining_years"]


def get_sorted_years(df: pd.DataFrame) -> list[int]:
    """
    取得資料中的年份，並由小到大排序。
    """
    if YEAR_COL not in df.columns:
        raise ValueError(f"資料缺少必要欄位：{YEAR_COL}")

    years = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())

    if len(years) < 2:
        raise ValueError("Temporal validation 至少需要兩個年份：一個 training year 與一個 testing year。")

    return years


def validate_year_distribution(
    df: pd.DataFrame,
    expected_rows_per_year: int | None = 200,
) -> pd.DataFrame:
    """
    檢查每年資料筆數。

    Parameters
    ----------
    df : pd.DataFrame
        清理後資料。
    expected_rows_per_year : int | None
        若指定數值，則檢查每年是否等於該筆數。
        對本專題原始資料，應為每年 200 筆。

    Returns
    -------
    pd.DataFrame
        每年資料筆數表。
    """
    year_counts = (
        df.groupby(YEAR_COL)
        .size()
        .reset_index(name="row_count")
        .sort_values(YEAR_COL)
        .reset_index(drop=True)
    )

    if expected_rows_per_year is not None:
        invalid = year_counts[year_counts["row_count"] != expected_rows_per_year]

        if not invalid.empty:
            raise ValueError(
                "發現部分年份筆數不符合預期。\n"
                f"Expected rows per year: {expected_rows_per_year}\n"
                f"{invalid.to_string(index=False)}"
            )

    return year_counts


def make_temporal_splits(
    df: pd.DataFrame,
    mode: ValidationMode = "next_year",
    min_train_years: int = 1,
) -> list[dict]:
    """
    建立 expanding-window temporal validation splits。

    mode = "next_year":
        Split 1: Train 1997 -> Test 1998
        Split 2: Train 1997-1998 -> Test 1999
        ...
        適合年度投資組合回測。

    mode = "remaining_years":
        TV1: Train 1997 -> Test 1998-2008
        TV2: Train 1997-1998 -> Test 1999-2008
        ...
        較貼近課堂 TV 圖示。

    Parameters
    ----------
    df : pd.DataFrame
        清理後資料，必須包含 year 欄位。
    mode : {"next_year", "remaining_years"}
        split 模式。
    min_train_years : int
        最少訓練年份數。預設 1，表示第一個 split 使用 1997 訓練、1998 測試。

    Returns
    -------
    list[dict]
        每個 split 的 train_years / test_years 等資訊。
    """
    if mode not in ["next_year", "remaining_years"]:
        raise ValueError(f"不支援的 validation mode：{mode}")

    years = get_sorted_years(df)

    if min_train_years < 1:
        raise ValueError("min_train_years 必須 >= 1")

    if min_train_years >= len(years):
        raise ValueError("min_train_years 必須小於總年份數，否則沒有 testing year。")

    splits = []

    split_id = 1

    for train_end_index in range(min_train_years - 1, len(years) - 1):
        train_years = years[: train_end_index + 1]

        if mode == "next_year":
            test_years = [years[train_end_index + 1]]
        else:
            test_years = years[train_end_index + 1 :]

        split = {
            "split_id": split_id,
            "mode": mode,
            "train_years": train_years,
            "test_years": test_years,
        }

        splits.append(split)
        split_id += 1

    return splits


def validate_no_future_leakage(splits: list[dict]) -> None:
    """
    確認每個 split 沒有 future data leakage。

    合法條件：
    1. train_years 與 test_years 不可重疊。
    2. max(train_years) 必須小於 min(test_years)。
    """
    for split in splits:
        split_id = split["split_id"]
        train_years = split["train_years"]
        test_years = split["test_years"]

        train_set = set(train_years)
        test_set = set(test_years)

        overlap = train_set.intersection(test_set)

        if overlap:
            raise ValueError(
                f"Split {split_id} 發現 training years 與 testing years 重疊：{sorted(overlap)}"
            )

        if max(train_years) >= min(test_years):
            raise ValueError(
                f"Split {split_id} 發現 future leakage："
                f"max(train_years)={max(train_years)} >= min(test_years)={min(test_years)}"
            )


def _count_label_values(df: pd.DataFrame) -> dict:
    """
    計算 label -1 與 label 1 的數量。
    """
    counts = df[TARGET_CLASS].value_counts().to_dict()

    return {
        "label_neg1_count": int(counts.get(-1, 0)),
        "label_pos1_count": int(counts.get(1, 0)),
    }


def summarize_splits(
    df: pd.DataFrame,
    splits: list[dict],
) -> pd.DataFrame:
    """
    將 split list 轉成可輸出 CSV 的摘要表。

    每列包含：
    - split_id
    - mode
    - train_years
    - test_years
    - train_rows
    - test_rows
    - train label distribution
    - test label distribution
    - leakage check status
    """
    records = []

    for split in splits:
        split_id = split["split_id"]
        mode = split["mode"]
        train_years = split["train_years"]
        test_years = split["test_years"]

        train_df = df[df[YEAR_COL].isin(train_years)].copy()
        test_df = df[df[YEAR_COL].isin(test_years)].copy()

        train_label_counts = _count_label_values(train_df)
        test_label_counts = _count_label_values(test_df)

        record = {
            "split_id": split_id,
            "mode": mode,
            "train_years": ",".join(map(str, train_years)),
            "test_years": ",".join(map(str, test_years)),
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "test_start_year": min(test_years),
            "test_end_year": max(test_years),
            "n_train_years": len(train_years),
            "n_test_years": len(test_years),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_label_neg1_count": train_label_counts["label_neg1_count"],
            "train_label_pos1_count": train_label_counts["label_pos1_count"],
            "test_label_neg1_count": test_label_counts["label_neg1_count"],
            "test_label_pos1_count": test_label_counts["label_pos1_count"],
            "max_train_year": max(train_years),
            "min_test_year": min(test_years),
            "leakage_free": max(train_years) < min(test_years),
        }

        records.append(record)

    summary_df = pd.DataFrame(records)

    if not summary_df["leakage_free"].all():
        invalid = summary_df[~summary_df["leakage_free"]]
        raise ValueError(
            "部分 split 未通過 leakage_free 檢查：\n"
            f"{invalid.to_string(index=False)}"
        )

    return summary_df


def generate_split_profile(
    df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
    mode: ValidationMode,
) -> str:
    """
    產生 split profile 文字，方便檢查與放進報告。
    """
    years = get_sorted_years(df)
    year_counts = validate_year_distribution(df, expected_rows_per_year=None)

    lines = []

    lines.append(f"========== Step 2 Temporal Validation Profile: {mode} ==========")
    lines.append("")
    lines.append("Dataset summary:")
    lines.append(f"- Total rows: {len(df)}")
    lines.append(f"- Year range: {min(years)} - {max(years)}")
    lines.append(f"- Number of years: {len(years)}")
    lines.append("")

    lines.append("Rows by year:")
    for _, row in year_counts.iterrows():
        lines.append(f"- {int(row[YEAR_COL])}: {int(row['row_count'])} rows")
    lines.append("")

    lines.append("Split summary:")
    lines.append(f"- Validation mode: {mode}")
    lines.append(f"- Number of splits: {len(split_summary_df)}")
    lines.append(f"- Leakage-free: {bool(split_summary_df['leakage_free'].all())}")
    lines.append("")

    lines.append("Detailed splits:")
    display_cols = [
        "split_id",
        "train_years",
        "test_years",
        "train_rows",
        "test_rows",
        "train_label_neg1_count",
        "train_label_pos1_count",
        "test_label_neg1_count",
        "test_label_pos1_count",
        "leakage_free",
    ]

    lines.append(split_summary_df[display_cols].to_string(index=False))
    lines.append("")

    if mode == "next_year":
        lines.append("Interpretation:")
        lines.append(
            "- This mode is used as the main portfolio backtesting design. "
            "Each testing year corresponds to one annual investment period."
        )
    elif mode == "remaining_years":
        lines.append("Interpretation:")
        lines.append(
            "- This mode follows the classroom TV diagram more closely. "
            "Each split trains on earlier years and evaluates on all remaining future years."
        )

    lines.append("")
    lines.append("========== End of Step 2 Profile ==========")

    return "\n".join(lines)


def save_split_summary(
    split_summary_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    儲存 split summary CSV。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_split_profile(
    profile_text: str,
    output_path: str | Path,
) -> None:
    """
    儲存 split profile 文字檔。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(profile_text)


def build_and_save_temporal_splits(
    df: pd.DataFrame,
    mode: ValidationMode,
    split_output_path: str | Path,
    profile_output_path: str | Path,
    expected_rows_per_year: int | None = 200,
) -> pd.DataFrame:
    """
    建立、檢查並儲存 temporal validation splits。
    """
    validate_year_distribution(df, expected_rows_per_year=expected_rows_per_year)

    splits = make_temporal_splits(df, mode=mode)
    validate_no_future_leakage(splits)

    split_summary_df = summarize_splits(df, splits)
    profile_text = generate_split_profile(df, split_summary_df, mode=mode)

    save_split_summary(split_summary_df, split_output_path)
    save_split_profile(profile_text, profile_output_path)

    return split_summary_df