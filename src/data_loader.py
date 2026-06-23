from pathlib import Path

import pandas as pd


def load_excel_data(file_path: str | Path) -> pd.DataFrame:
    """
    讀取 Excel 資料。
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到資料檔案：{file_path}")

    if file_path.suffix.lower() not in [".xlsx", ".xls"]:
        raise ValueError(f"檔案格式錯誤，請使用 Excel 檔案：{file_path}")

    df = pd.read_excel(file_path)

    if df.empty:
        raise ValueError(f"資料檔案為空：{file_path}")

    return df


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    儲存 DataFrame 為 CSV。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")