from pathlib import Path

from PIL import Image

from src.config import (
    STEP6_ALL_MODEL_TOP10_CUMULATIVE_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOP10_ANNUAL_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOPK_HEATMAP_FIG_PATH,
    STEP6_ALL_MODEL_RETURN_RISK_SCATTER_FIG_PATH,
    STEP6_ALL_MODEL_SHARPE_FIG_PATH,
    STEP6_ALL_MODEL_DRAWDOWN_FIG_PATH,
    STEP6_SVR_PREDICTED_VS_ACTUAL_FIG_PATH,
)


REQUIRED_PNGS = [
    STEP6_ALL_MODEL_TOP10_CUMULATIVE_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOP10_ANNUAL_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOPK_HEATMAP_FIG_PATH,
    STEP6_ALL_MODEL_RETURN_RISK_SCATTER_FIG_PATH,
    STEP6_ALL_MODEL_SHARPE_FIG_PATH,
    STEP6_ALL_MODEL_DRAWDOWN_FIG_PATH,
    STEP6_SVR_PREDICTED_VS_ACTUAL_FIG_PATH,
]


def validate_png(path: str | Path) -> None:
    path = Path(path)

    assert path.exists(), f"PNG 不存在：{path}"
    assert path.suffix.lower() == ".png", f"檔案副檔名不是 .png：{path}"
    assert path.stat().st_size > 5_000, f"PNG 檔案過小，可能輸出失敗：{path}"

    with Image.open(path) as img:
        img.verify()

    with Image.open(path) as img:
        width, height = img.size

    assert width >= 400, f"PNG 寬度過小：{path}, width={width}"
    assert height >= 300, f"PNG 高度過小：{path}, height={height}"

    print(f"[Pass] {path} | size={width}x{height} | bytes={path.stat().st_size}")


def main() -> None:
    print("========== Step 6 All-model PNG Validation ==========")

    for png_path in REQUIRED_PNGS:
        validate_png(png_path)

    print("")
    print("[Pass] All Step 6 all-model PNG files are valid.")
    print("========== Step 6 All-model PNG Validation Finished ==========")


if __name__ == "__main__":
    main()