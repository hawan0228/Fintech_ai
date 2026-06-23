# step7_build_external_metadata.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    EXTERNAL_TICKERS_PATH,
    EXTERNAL_CLEANED_DATA_PATH,
    EXTERNAL_FEATURE_MAPPING_PATH,
    EXTERNAL_UNIVERSE_PATH,
    EXTERNAL_DATA_QUALITY_REPORT_PATH,
    EXTERNAL_DATA_QUALITY_SUMMARY_PATH,
    EXTERNAL_CRAWLER_LOG_PATH,
)
from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEAR_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_feature_mapping(df: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = df[FEATURE_COLUMNS].isna().mean().to_dict()

    records = [
        {
            "original_feature": "市值(百萬元)",
            "external_feature_name": "market_cap_million",
            "source": "FinMind BalanceSheet + FinMind/yfinance price",
            "formula": "year-end close price × estimated shares outstanding / 1,000,000",
            "available": "yes" if missing_ratio["市值(百萬元)"] < 1 else "no",
            "missing_ratio": missing_ratio["市值(百萬元)"],
            "used_in_model": "yes" if missing_ratio["市值(百萬元)"] < 1 else "no",
            "limitation": "Shares outstanding are estimated from share capital and par value; may differ from official market capitalization.",
        },
        {
            "original_feature": "收盤價(元)_年",
            "external_feature_name": "year-end anchor close price",
            "source": "FinMind TaiwanStockPrice / yfinance fallback",
            "formula": "first available trading close price in December",
            "available": "yes" if missing_ratio["收盤價(元)_年"] < 1 else "no",
            "missing_ratio": missing_ratio["收盤價(元)_年"],
            "used_in_model": "yes" if missing_ratio["收盤價(元)_年"] < 1 else "no",
            "limitation": "Uses December anchor price, which approximates the original December price definition.",
        },
        {
            "original_feature": "Unknown masked parameter",
            "external_feature_name": "PER proxy",
            "source": "FinMind TaiwanStockPER",
            "formula": "year-end PER-like valuation proxy",
            "available": "yes" if missing_ratio["Unknown masked parameter"] < 1 else "no",
            "missing_ratio": missing_ratio["Unknown masked parameter"],
            "used_in_model": "yes" if missing_ratio["Unknown masked parameter"] < 1 else "no",
            "limitation": "Original variable is masked; external value is treated as a valuation proxy.",
        },
        {
            "original_feature": "股價淨值比",
            "external_feature_name": "PBR",
            "source": "FinMind TaiwanStockPER",
            "formula": "year-end price-to-book ratio",
            "available": "yes" if missing_ratio["股價淨值比"] < 1 else "no",
            "missing_ratio": missing_ratio["股價淨值比"],
            "used_in_model": "yes" if missing_ratio["股價淨值比"] < 1 else "no",
            "limitation": "Uses available year-end PBR proxy.",
        },
        {
            "original_feature": "股價營收比",
            "external_feature_name": "price_to_sales_proxy",
            "source": "FinMind financial statements + price",
            "formula": "market capitalization / annual revenue",
            "available": "yes" if missing_ratio["股價營收比"] < 1 else "no",
            "missing_ratio": missing_ratio["股價營收比"],
            "used_in_model": "yes" if missing_ratio["股價營收比"] < 1 else "no",
            "limitation": "Depends on revenue and market capitalization availability.",
        },
        {
            "original_feature": "淨值報酬率─稅後",
            "external_feature_name": "ROE_proxy",
            "source": "FinMind FinancialStatements + BalanceSheet",
            "formula": "net income / average total equity × 100",
            "available": "yes" if missing_ratio["淨值報酬率─稅後"] < 1 else "no",
            "missing_ratio": missing_ratio["淨值報酬率─稅後"],
            "used_in_model": "yes" if missing_ratio["淨值報酬率─稅後"] < 1 else "no",
            "limitation": "Computed proxy; may differ from TEJ or course dataset definition.",
        },
        {
            "original_feature": "資產報酬率 ROA",
            "external_feature_name": "ROA_proxy",
            "source": "FinMind FinancialStatements + BalanceSheet",
            "formula": "net income / average total assets × 100",
            "available": "yes" if missing_ratio["資產報酬率 ROA"] < 1 else "no",
            "missing_ratio": missing_ratio["資產報酬率 ROA"],
            "used_in_model": "yes" if missing_ratio["資產報酬率 ROA"] < 1 else "no",
            "limitation": "Computed proxy; dependent on balance sheet item matching.",
        },
        {
            "original_feature": "營業利益率 OPM",
            "external_feature_name": "OPM_proxy",
            "source": "FinMind FinancialStatements",
            "formula": "operating income / revenue × 100",
            "available": "yes" if missing_ratio["營業利益率 OPM"] < 1 else "no",
            "missing_ratio": missing_ratio["營業利益率 OPM"],
            "used_in_model": "yes" if missing_ratio["營業利益率 OPM"] < 1 else "no",
            "limitation": "Computed proxy; dependent on income statement item matching.",
        },
        {
            "original_feature": "利潤邊際 NPM",
            "external_feature_name": "NPM_proxy",
            "source": "FinMind FinancialStatements",
            "formula": "net income / revenue × 100",
            "available": "yes" if missing_ratio["利潤邊際 NPM"] < 1 else "no",
            "missing_ratio": missing_ratio["利潤邊際 NPM"],
            "used_in_model": "yes" if missing_ratio["利潤邊際 NPM"] < 1 else "no",
            "limitation": "Computed proxy; dependent on income statement item matching.",
        },
        {
            "original_feature": "營業利益成長率",
            "external_feature_name": "revenue_growth_proxy / operating growth proxy",
            "source": "FinMind MonthRevenue or FinancialStatements",
            "formula": "annual revenue growth or operating income growth",
            "available": "yes" if missing_ratio["營業利益成長率"] < 1 else "partial",
            "missing_ratio": missing_ratio["營業利益成長率"],
            "used_in_model": "yes" if missing_ratio["營業利益成長率"] < 1 else "no",
            "limitation": "Currently may use revenue growth proxy when operating income growth is unavailable.",
        },
        {
            "original_feature": "稅後淨利成長率",
            "external_feature_name": "net_income_growth_proxy",
            "source": "FinMind FinancialStatements",
            "formula": "(net income_t - net income_t-1) / |net income_t-1| × 100",
            "available": "yes" if missing_ratio["稅後淨利成長率"] < 1 else "no",
            "missing_ratio": missing_ratio["稅後淨利成長率"],
            "used_in_model": "yes" if missing_ratio["稅後淨利成長率"] < 1 else "no",
            "limitation": "First year per stock is missing because YoY growth requires prior-year net income.",
        },
    ]

    covered = {r["original_feature"] for r in records}

    for feature in FEATURE_COLUMNS:
        if feature not in covered:
            records.append(
                {
                    "original_feature": feature,
                    "external_feature_name": "",
                    "source": "",
                    "formula": "",
                    "available": "yes" if missing_ratio[feature] < 1 else "no",
                    "missing_ratio": missing_ratio[feature],
                    "used_in_model": "yes" if missing_ratio[feature] < 1 else "no",
                    "limitation": "Not yet reconstructed from external data source.",
                }
            )

    mapping_df = pd.DataFrame(records)
    mapping_df = mapping_df.sort_values("missing_ratio", ascending=False).reset_index(drop=True)

    return mapping_df


def build_external_universe(df: pd.DataFrame) -> pd.DataFrame:
    tickers_df = pd.read_csv(EXTERNAL_TICKERS_PATH, dtype={"stock_id": str})
    tickers_df["stock_id"] = tickers_df["stock_id"].astype(str).str.zfill(4)

    if "stock_name" not in tickers_df.columns:
        tickers_df["stock_name"] = tickers_df["stock_id"]

    rows = []

    for stock_id, group_df in df.groupby(STOCK_ID_COL):
        group_df = group_df.copy()

        stock_name = str(group_df[STOCK_NAME_COL].iloc[0])
        years = sorted(group_df[YEAR_COL].astype(int).unique().tolist())

        rows.append(
            {
                "stock_id": stock_id,
                "stock_name": stock_name,
                "market": "",
                "industry": "",
                "source": "data/external/tickers.csv",
                "inclusion_reason": "Selected as external universe constituent with available public data.",
                "first_available_year": min(years),
                "last_available_year": max(years),
                "n_years": len(years),
                "n_rows": len(group_df),
            }
        )

    universe_df = pd.DataFrame(rows)

    return universe_df.sort_values("stock_id").reset_index(drop=True)


def build_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    # Dataset-level checks
    records.append(
        {
            "section": "dataset",
            "item": "total_rows",
            "value": len(df),
            "note": "",
        }
    )

    records.append(
        {
            "section": "dataset",
            "item": "n_unique_stocks",
            "value": df[STOCK_ID_COL].nunique(),
            "note": "",
        }
    )

    records.append(
        {
            "section": "dataset",
            "item": "n_years",
            "value": df[YEAR_COL].nunique(),
            "note": "",
        }
    )

    duplicate_count = df.duplicated([STOCK_ID_COL, YEAR_COL]).sum()
    records.append(
        {
            "section": "data_integrity",
            "item": "duplicate_stock_year_count",
            "value": int(duplicate_count),
            "note": "Should be 0.",
        }
    )

    invalid_return_count = (df[TARGET_RETURN] < -100).sum()
    records.append(
        {
            "section": "data_integrity",
            "item": "return_less_than_minus_100_count",
            "value": int(invalid_return_count),
            "note": "Return should generally be >= -100.",
        }
    )

    # Rows by year
    for year, count in df.groupby(YEAR_COL).size().items():
        records.append(
            {
                "section": "rows_by_year",
                "item": int(year),
                "value": int(count),
                "note": "",
            }
        )

    # Label distribution by year
    label_counts = (
        df.groupby([YEAR_COL, TARGET_CLASS])
        .size()
        .reset_index(name="count")
    )

    for _, row in label_counts.iterrows():
        records.append(
            {
                "section": "label_distribution_by_year",
                "item": f"year={int(row[YEAR_COL])}, label={int(row[TARGET_CLASS])}",
                "value": int(row["count"]),
                "note": "",
            }
        )

    # Return summary by year
    return_summary = (
        df.groupby(YEAR_COL)[TARGET_RETURN]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    for _, row in return_summary.iterrows():
        for metric in ["mean", "std", "min", "max"]:
            records.append(
                {
                    "section": "return_summary_by_year",
                    "item": f"year={int(row[YEAR_COL])}, {metric}",
                    "value": row[metric],
                    "note": "",
                }
            )

    # Feature missing ratio
    for feature in FEATURE_COLUMNS:
        records.append(
            {
                "section": "feature_missing_ratio",
                "item": feature,
                "value": df[feature].isna().mean(),
                "note": "",
            }
        )

    # Crawler log source success rate
    if Path(EXTERNAL_CRAWLER_LOG_PATH).exists():
        log_df = pd.read_csv(EXTERNAL_CRAWLER_LOG_PATH)

        if not log_df.empty and "dataset" in log_df.columns:
            source_success = (
                log_df.groupby(["source", "dataset"])["success"]
                .mean()
                .reset_index(name="success_rate")
            )

            for _, row in source_success.iterrows():
                records.append(
                    {
                        "section": "crawler_success_rate",
                        "item": f"{row['source']}:{row['dataset']}",
                        "value": row["success_rate"],
                        "note": "",
                    }
                )

    return pd.DataFrame(records)


def build_summary_text(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("========== External Data Quality Summary ==========")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Unique stocks: {df[STOCK_ID_COL].nunique()}")
    lines.append(f"- Years: {df[YEAR_COL].min()} - {df[YEAR_COL].max()}")
    lines.append("")
    lines.append("Feature missing ratio:")
    missing_ratio = df[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    lines.append(missing_ratio.to_string())
    lines.append("")
    lines.append("External universe:")
    lines.append(universe_df.head(20).to_string(index=False))
    lines.append("")
    lines.append("Feature mapping:")
    lines.append(
        mapping_df[
            [
                "original_feature",
                "external_feature_name",
                "source",
                "formula",
                "available",
                "missing_ratio",
                "used_in_model",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("Integrity checks:")
    check_df = quality_df[
        quality_df["section"].isin(["dataset", "data_integrity"])
    ]
    lines.append(check_df.to_string(index=False))
    lines.append("")
    lines.append("========== End Summary ==========")

    return "\n".join(lines)


def main() -> None:
    print("========== Step 7: Build External Metadata and Quality Reports ==========")

    df = pd.read_csv(EXTERNAL_CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_RETURN] = pd.to_numeric(df[TARGET_RETURN], errors="coerce")
    df[TARGET_CLASS] = pd.to_numeric(df[TARGET_CLASS], errors="coerce")

    mapping_df = build_feature_mapping(df)
    universe_df = build_external_universe(df)
    quality_df = build_data_quality_report(df)

    summary_text = build_summary_text(
        df=df,
        mapping_df=mapping_df,
        universe_df=universe_df,
        quality_df=quality_df,
    )

    save_dataframe(mapping_df, EXTERNAL_FEATURE_MAPPING_PATH)
    save_dataframe(universe_df, EXTERNAL_UNIVERSE_PATH)
    save_dataframe(quality_df, EXTERNAL_DATA_QUALITY_REPORT_PATH)
    save_text(summary_text, EXTERNAL_DATA_QUALITY_SUMMARY_PATH)

    print(summary_text)
    print("")
    print("[Saved]")
    print(f"- {EXTERNAL_FEATURE_MAPPING_PATH}")
    print(f"- {EXTERNAL_UNIVERSE_PATH}")
    print(f"- {EXTERNAL_DATA_QUALITY_REPORT_PATH}")
    print(f"- {EXTERNAL_DATA_QUALITY_SUMMARY_PATH}")

    print("========== External Metadata Finished ==========")


if __name__ == "__main__":
    main()