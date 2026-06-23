import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
    TEMPORAL_SPLITS_REMAINING_YEARS_PATH,
    SPLIT_PROFILE_NEXT_YEAR_PATH,
    SPLIT_PROFILE_REMAINING_YEARS_PATH,
)
from src.schema import STOCK_ID_COL
from src.validation import build_and_save_temporal_splits


def main() -> None:
    print("========== Temporal Validation Split ==========")

    print(f"[Info] Loading cleaned data from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})

    print(f"[Info] Cleaned data shape: {df.shape}")

    print("")
    print("[Info] Building next_year temporal splits...")
    next_year_summary = build_and_save_temporal_splits(
        df=df,
        mode="next_year",
        split_output_path=TEMPORAL_SPLITS_NEXT_YEAR_PATH,
        profile_output_path=SPLIT_PROFILE_NEXT_YEAR_PATH,
        expected_rows_per_year=200,
    )

    print(f"[Info] Saved next_year splits to: {TEMPORAL_SPLITS_NEXT_YEAR_PATH}")
    print(f"[Info] Saved next_year profile to: {SPLIT_PROFILE_NEXT_YEAR_PATH}")
    print("")
    print(next_year_summary.to_string(index=False))

    print("")
    print("[Info] Building remaining_years temporal splits...")
    remaining_years_summary = build_and_save_temporal_splits(
        df=df,
        mode="remaining_years",
        split_output_path=TEMPORAL_SPLITS_REMAINING_YEARS_PATH,
        profile_output_path=SPLIT_PROFILE_REMAINING_YEARS_PATH,
        expected_rows_per_year=200,
    )

    print(f"[Info] Saved remaining_years splits to: {TEMPORAL_SPLITS_REMAINING_YEARS_PATH}")
    print(f"[Info] Saved remaining_years profile to: {SPLIT_PROFILE_REMAINING_YEARS_PATH}")
    print("")
    print(remaining_years_summary.to_string(index=False))

    print("")
    print("========== Finished Successfully ==========")


if __name__ == "__main__":
    main()