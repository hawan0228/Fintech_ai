from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, DATA_PROFILE_PATH
from src.data_loader import load_excel_data, save_dataframe
from src.preprocessing import clean_raw_data, generate_data_profile, save_data_profile


def main() -> None:
    print("========== Data Loading generate_data_profile, save_data_profile")


def main() -> None:
    print("========== Data Loading and Cleaning ==========")

    print(f"[Info] Loading raw data from: {RAW_DATA_PATH}")
    raw_df = load_excel_data(RAW_DATA_PATH)

    print(f"[Info] Raw data shape: {raw_df.shape}")

    print("[Info] Cleaning raw data...")
    cleaned_df = clean_raw_data(raw_df)

    print(f"[Info] Cleaned data shape: {cleaned_df.shape}")

    print(f"[Info] Saving cleaned data to: {CLEANED_DATA_PATH}")
    save_dataframe(cleaned_df, CLEANED_DATA_PATH)

    print(f"[Info] Generating data profile: {DATA_PROFILE_PATH}")
    profile_text = generate_data_profile(cleaned_df)
    save_data_profile(profile_text, DATA_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")


if __name__ == "__main__":
    main()