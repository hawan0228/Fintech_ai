import pandas as pd

from src.config import (
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
    TEMPORAL_SPLITS_REMAINING_YEARS_PATH,
)
from src.schema import YEAR_COL


def parse_years(years_text: str) -> list[int]:
    """
    將 '1997,1998,1999' 轉成 [1997, 1998, 1999]。
    """
    return [int(x) for x in str(years_text).split(",") if str(x).strip()]


def validate_split_file(path, mode: str) -> None:
    print("")
    print(f"========== Validating {mode} splits ==========")
    print(f"[Info] Loading: {path}")

    df = pd.read_csv(path)

    print(f"[Info] Shape: {df.shape}")

    assert len(df) == 11, f"{mode} 應有 11 個 split，但目前是 {len(df)}"

    required_columns = [
        "split_id",
        "mode",
        "train_years",
        "test_years",
        "train_rows",
        "test_rows",
        "leakage_free",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    assert not missing, f"{mode} split 檔案缺少欄位：{missing}"

    assert df["leakage_free"].astype(bool).all(), f"{mode} 存在 leakage_free=False 的 split"

    for _, row in df.iterrows():
        split_id = int(row["split_id"])
        train_years = parse_years(row["train_years"])
        test_years = parse_years(row["test_years"])

        # 1. training / testing 不可重疊
        overlap = set(train_years).intersection(set(test_years))
        assert not overlap, f"{mode} Split {split_id} train/test years 重疊：{overlap}"

        # 2. 不可使用未來資料
        assert max(train_years) < min(test_years), (
            f"{mode} Split {split_id} 發生 future leakage："
            f"max(train_years)={max(train_years)}, min(test_years)={min(test_years)}"
        )

        # 3. train rows 檢查
        expected_train_rows = len(train_years) * 200
        expected_test_rows = len(test_years) * 200

        assert int(row["train_rows"]) == expected_train_rows, (
            f"{mode} Split {split_id} train_rows 應為 {expected_train_rows}，"
            f"但目前是 {row['train_rows']}"
        )

        assert int(row["test_rows"]) == expected_test_rows, (
            f"{mode} Split {split_id} test_rows 應為 {expected_test_rows}，"
            f"但目前是 {row['test_rows']}"
        )

        # 4. next_year 模式 testing year 應只有一年
        if mode == "next_year":
            assert len(test_years) == 1, (
                f"next_year Split {split_id} testing year 應只有 1 年，"
                f"但目前是 {test_years}"
            )

        # 5. remaining_years 模式 testing years 應為所有剩餘未來年份
        if mode == "remaining_years":
            expected_test_start = max(train_years) + 1
            assert min(test_years) == expected_test_start, (
                f"remaining_years Split {split_id} testing start year 應為 {expected_test_start}，"
                f"但目前是 {min(test_years)}"
            )

    print(f"[Pass] {mode} splits validation passed.")


def main():
    print("========== Step 2 Final Validation ==========")

    validate_split_file(TEMPORAL_SPLITS_NEXT_YEAR_PATH, mode="next_year")
    validate_split_file(TEMPORAL_SPLITS_REMAINING_YEARS_PATH, mode="remaining_years")

    print("")
    print("[Pass] All Step 2 split files are valid.")
    print("========== Step 2 Validation Finished ==========")


if __name__ == "__main__":
    main()