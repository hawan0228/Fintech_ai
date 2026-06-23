# step7_run_external_crawler.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    EXTERNAL_TICKERS_PATH,
    EXTERNAL_CLEANED_DATA_PATH,
    EXTERNAL_PROFILE_PATH,
    EXTERNAL_CRAWLER_LOG_PATH,
)
from src.schema import STOCK_ID_COL, STOCK_NAME_COL, YEAR_COL
from src.external.external_config import (
    DEFAULT_EXTERNAL_START_YEAR,
    DEFAULT_EXTERNAL_END_YEAR,
    EXTERNAL_RAW_DIR,
    EXTERNAL_RAW_PRICE_DIR,
    EXTERNAL_RAW_PER_DIR,
    EXTERNAL_RAW_FINANCIAL_DIR,
    EXTERNAL_RAW_REVENUE_DIR,
)
from src.external.finmind_client import (
    FinMindRateLimitError,
    fetch_finmind_stock_price,
    fetch_finmind_per_pbr,
    fetch_finmind_month_revenue,
    fetch_finmind_financial_statement,
    fetch_finmind_balance_sheet,
    resolve_finmind_token,
    is_probably_valid_jwt,
    save_raw_df,
)
from src.external.yfinance_client import fetch_yfinance_price
from src.external.build_external_dataset import (
    build_anchor_prices,
    build_returns_from_anchor_prices,
    extract_year_end_per_pbr,
    extract_revenue_growth,
    extract_financial_statement_features,
    merge_external_features,
    generate_external_profile,
)
from src.external.financial_feature_engineering import (
    extract_income_statement_items,
    extract_balance_sheet_items,
    calculate_external_financial_features,
)


# =========================
# Paths
# =========================

EXTERNAL_RAW_BALANCE_DIR = EXTERNAL_RAW_DIR / "balance_sheet"
EXTERNAL_RAW_ENGINEERED_DIR = EXTERNAL_RAW_DIR / "engineered"


# =========================
# Request budget
# =========================

class RequestBudget:
    """
    控制每次執行最多新增多少 FinMind request。

    重點：
    - cache hit 不消耗 request budget
    - yfinance fallback 不消耗 FinMind request budget
    - 避免一次 60 檔 × 5 datasets 直接打爆額度
    """

    def __init__(self, max_new_requests: int | None = None):
        self.max_new_requests = max_new_requests
        self.used_new_requests = 0

    def can_request(self) -> bool:
        if self.max_new_requests is None:
            return True
        return self.used_new_requests < self.max_new_requests

    def consume(self) -> None:
        self.used_new_requests += 1

    def status(self) -> str:
        if self.max_new_requests is None:
            return f"{self.used_new_requests}/unlimited"
        return f"{self.used_new_requests}/{self.max_new_requests}"


# =========================
# Basic IO utilities
# =========================

def ensure_external_dirs() -> None:
    directories = [
        EXTERNAL_RAW_DIR,
        EXTERNAL_RAW_PRICE_DIR,
        EXTERNAL_RAW_PER_DIR,
        EXTERNAL_RAW_FINANCIAL_DIR,
        EXTERNAL_RAW_REVENUE_DIR,
        EXTERNAL_RAW_BALANCE_DIR,
        EXTERNAL_RAW_ENGINEERED_DIR,
        EXTERNAL_CLEANED_DATA_PATH.parent,
        EXTERNAL_PROFILE_PATH.parent,
        EXTERNAL_CRAWLER_LOG_PATH.parent,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def clean_stock_id(value) -> str:
    text = str(value).strip()
    text = text.replace(".0", "")

    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)

    return text


def read_tickers(path: str | Path, max_tickers: int | None = None) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到 tickers.csv：{path}\n"
            "請建立 data/external/tickers.csv，欄位至少包含 stock_id，可選 stock_name。"
        )

    df = pd.read_csv(path, dtype={"stock_id": str})

    if "stock_id" not in df.columns:
        raise ValueError("tickers.csv 必須包含 stock_id 欄位。")

    if "stock_name" not in df.columns:
        df["stock_name"] = df["stock_id"]

    df["stock_id"] = df["stock_id"].astype(str).map(clean_stock_id)
    df["stock_name"] = df["stock_name"].astype(str)

    out = df[["stock_id", "stock_name"]].drop_duplicates().reset_index(drop=True)

    if max_tickers is not None:
        out = out.head(int(max_tickers)).copy()

    if out.empty:
        raise ValueError("tickers.csv 沒有任何有效股票。")

    return out


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_cache_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, dtype={"stock_id": str})
    except Exception:
        return pd.DataFrame()


# =========================
# Fetch utilities
# =========================

def safe_fetch_with_cache(
    fetch_func,
    dataset_name: str,
    stock_id: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    request_budget: RequestBudget,
    token: str | None = None,
    use_token: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, str, str]:
    """
    Returns
    -------
    df, error_message, fetch_status

    fetch_status:
    - cache_hit
    - fetched
    - empty
    - skipped_budget
    - rate_limited
    - failed
    """
    output_path = Path(output_path)

    if use_cache and not force_refresh:
        cached_df = load_cache_if_exists(output_path)
        if not cached_df.empty:
            cached_df["stock_id"] = stock_id
            return cached_df, "", "cache_hit"

    if not request_budget.can_request():
        return (
            pd.DataFrame(),
            f"{dataset_name}: skipped because request budget reached",
            "skipped_budget",
        )

    try:
        request_budget.consume()

        if use_token:
            df = fetch_func(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
                token=token,
                verbose=verbose,
            )
        else:
            df = fetch_func(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )

        if df is None or df.empty:
            return pd.DataFrame(), f"{dataset_name}: empty response", "empty"

        df = df.copy()
        df["stock_id"] = stock_id

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_raw_df(df, output_path)

        return df, "", "fetched"

    except FinMindRateLimitError as exc:
        return pd.DataFrame(), f"{dataset_name}: {exc}", "rate_limited"

    except Exception as exc:
        return pd.DataFrame(), f"{dataset_name}: {type(exc).__name__}: {exc}", "failed"


def append_crawler_log(
    logs: list[dict],
    stock_id: str,
    stock_name: str,
    source: str,
    dataset: str,
    start_date: str,
    end_date: str,
    df: pd.DataFrame,
    error_message: str,
    saved_path: str | Path,
    fetch_status: str,
    request_budget_status: str,
) -> None:
    logs.append(
        {
            "stock_id": stock_id,
            "stock_name": stock_name,
            "source": source,
            "dataset": dataset,
            "start_date": start_date,
            "end_date": end_date,
            "success": bool(df is not None and not df.empty),
            "n_rows": int(len(df)) if df is not None else 0,
            "fetch_status": fetch_status,
            "request_budget_status": request_budget_status,
            "error_message": error_message,
            "saved_path": str(saved_path),
        }
    )


def fetch_finmind_dataset_and_log(
    *,
    fetch_func,
    dataset_name: str,
    stock_id: str,
    stock_name: str,
    source_name: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    token: str | None,
    request_budget: RequestBudget,
    crawler_logs: list[dict],
    use_cache: bool,
    force_refresh: bool,
    verbose_api: bool,
) -> tuple[pd.DataFrame, str, str]:
    df, error_message, fetch_status = safe_fetch_with_cache(
        fetch_func=fetch_func,
        dataset_name=dataset_name,
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        request_budget=request_budget,
        token=token,
        use_token=True,
        use_cache=use_cache,
        force_refresh=force_refresh,
        verbose=verbose_api,
    )

    append_crawler_log(
        logs=crawler_logs,
        stock_id=stock_id,
        stock_name=stock_name,
        source=source_name,
        dataset=dataset_name,
        start_date=start_date,
        end_date=end_date,
        df=df,
        error_message=error_message,
        saved_path=output_path,
        fetch_status=fetch_status,
        request_budget_status=request_budget.status(),
    )

    return df, error_message, fetch_status


def fetch_yfinance_price_and_log(
    *,
    stock_id: str,
    stock_name: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    crawler_logs: list[dict],
    use_cache: bool,
    force_refresh: bool,
) -> tuple[pd.DataFrame, str, str]:
    unlimited_budget = RequestBudget(max_new_requests=None)

    df, error_message, fetch_status = safe_fetch_with_cache(
        fetch_func=fetch_yfinance_price,
        dataset_name="price",
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        request_budget=unlimited_budget,
        token=None,
        use_token=False,
        use_cache=use_cache,
        force_refresh=force_refresh,
        verbose=False,
    )

    append_crawler_log(
        logs=crawler_logs,
        stock_id=stock_id,
        stock_name=stock_name,
        source="yfinance_fallback",
        dataset="price",
        start_date=start_date,
        end_date=end_date,
        df=df,
        error_message=error_message,
        saved_path=output_path,
        fetch_status=fetch_status,
        request_budget_status="yfinance_not_counted",
    )

    return df, error_message, fetch_status


def append_not_fetched_log(
    *,
    crawler_logs: list[dict],
    stock_id: str,
    stock_name: str,
    dataset_name: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    reason: str,
) -> None:
    append_crawler_log(
        logs=crawler_logs,
        stock_id=stock_id,
        stock_name=stock_name,
        source="not_fetched",
        dataset=dataset_name,
        start_date=start_date,
        end_date=end_date,
        df=pd.DataFrame(),
        error_message=reason,
        saved_path=output_path,
        fetch_status="not_supported",
        request_budget_status="not_applicable",
    )


def fetch_one_stock(
    *,
    stock_id: str,
    stock_name: str,
    source: str,
    start_date: str,
    end_date: str,
    token: str | None,
    crawler_logs: list[dict],
    request_budget: RequestBudget,
    use_cache: bool,
    force_refresh: bool,
    stop_on_limit: bool,
    verbose_api: bool,
) -> dict[str, pd.DataFrame]:
    """
    抓取單一股票的所有外部資料。

    FinMind:
    - price, PER/PBR, month revenue, financial statement, balance sheet

    yfinance:
    - price only
    """
    print(f"\n[Info] Fetching {stock_id} {stock_name}")

    price_path = EXTERNAL_RAW_PRICE_DIR / f"{stock_id}_price.csv"
    per_path = EXTERNAL_RAW_PER_DIR / f"{stock_id}_per_pbr.csv"
    revenue_path = EXTERNAL_RAW_REVENUE_DIR / f"{stock_id}_revenue.csv"
    financial_path = EXTERNAL_RAW_FINANCIAL_DIR / f"{stock_id}_financial.csv"
    balance_path = EXTERNAL_RAW_BALANCE_DIR / f"{stock_id}_balance_sheet.csv"

    price_df = pd.DataFrame()
    per_df = pd.DataFrame()
    revenue_df = pd.DataFrame()
    financial_df = pd.DataFrame()
    balance_df = pd.DataFrame()

    if source == "finmind":
        price_df, price_error, price_status = fetch_finmind_dataset_and_log(
            fetch_func=fetch_finmind_stock_price,
            dataset_name="price",
            stock_id=stock_id,
            stock_name=stock_name,
            source_name="finmind",
            start_date=start_date,
            end_date=end_date,
            output_path=price_path,
            token=token,
            request_budget=request_budget,
            crawler_logs=crawler_logs,
            use_cache=use_cache,
            force_refresh=force_refresh,
            verbose_api=verbose_api,
        )

        if price_status == "rate_limited":
            if stop_on_limit:
                raise FinMindRateLimitError(price_error)
        elif price_df.empty:
            print(
                "[Warning] FinMind price unavailable. "
                "Trying yfinance fallback for price only..."
            )

            price_df, _, _ = fetch_yfinance_price_and_log(
                stock_id=stock_id,
                stock_name=stock_name,
                start_date=start_date,
                end_date=end_date,
                output_path=price_path,
                crawler_logs=crawler_logs,
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        per_df, per_error, per_status = fetch_finmind_dataset_and_log(
            fetch_func=fetch_finmind_per_pbr,
            dataset_name="per_pbr",
            stock_id=stock_id,
            stock_name=stock_name,
            source_name="finmind",
            start_date=start_date,
            end_date=end_date,
            output_path=per_path,
            token=token,
            request_budget=request_budget,
            crawler_logs=crawler_logs,
            use_cache=use_cache,
            force_refresh=force_refresh,
            verbose_api=verbose_api,
        )

        if per_status == "rate_limited" and stop_on_limit:
            raise FinMindRateLimitError(per_error)

        revenue_df, revenue_error, revenue_status = fetch_finmind_dataset_and_log(
            fetch_func=fetch_finmind_month_revenue,
            dataset_name="month_revenue",
            stock_id=stock_id,
            stock_name=stock_name,
            source_name="finmind",
            start_date=start_date,
            end_date=end_date,
            output_path=revenue_path,
            token=token,
            request_budget=request_budget,
            crawler_logs=crawler_logs,
            use_cache=use_cache,
            force_refresh=force_refresh,
            verbose_api=verbose_api,
        )

        if revenue_status == "rate_limited" and stop_on_limit:
            raise FinMindRateLimitError(revenue_error)

        financial_df, financial_error, financial_status = fetch_finmind_dataset_and_log(
            fetch_func=fetch_finmind_financial_statement,
            dataset_name="financial_statement",
            stock_id=stock_id,
            stock_name=stock_name,
            source_name="finmind",
            start_date=start_date,
            end_date=end_date,
            output_path=financial_path,
            token=token,
            request_budget=request_budget,
            crawler_logs=crawler_logs,
            use_cache=use_cache,
            force_refresh=force_refresh,
            verbose_api=verbose_api,
        )

        if financial_status == "rate_limited" and stop_on_limit:
            raise FinMindRateLimitError(financial_error)

        balance_df, balance_error, balance_status = fetch_finmind_dataset_and_log(
            fetch_func=fetch_finmind_balance_sheet,
            dataset_name="balance_sheet",
            stock_id=stock_id,
            stock_name=stock_name,
            source_name="finmind",
            start_date=start_date,
            end_date=end_date,
            output_path=balance_path,
            token=token,
            request_budget=request_budget,
            crawler_logs=crawler_logs,
            use_cache=use_cache,
            force_refresh=force_refresh,
            verbose_api=verbose_api,
        )

        if balance_status == "rate_limited" and stop_on_limit:
            raise FinMindRateLimitError(balance_error)

    else:
        price_df, _, _ = fetch_yfinance_price_and_log(
            stock_id=stock_id,
            stock_name=stock_name,
            start_date=start_date,
            end_date=end_date,
            output_path=price_path,
            crawler_logs=crawler_logs,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

        append_not_fetched_log(
            crawler_logs=crawler_logs,
            stock_id=stock_id,
            stock_name=stock_name,
            dataset_name="per_pbr",
            start_date=start_date,
            end_date=end_date,
            output_path=per_path,
            reason="source=yfinance; per_pbr not fetched",
        )
        append_not_fetched_log(
            crawler_logs=crawler_logs,
            stock_id=stock_id,
            stock_name=stock_name,
            dataset_name="month_revenue",
            start_date=start_date,
            end_date=end_date,
            output_path=revenue_path,
            reason="source=yfinance; month_revenue not fetched",
        )
        append_not_fetched_log(
            crawler_logs=crawler_logs,
            stock_id=stock_id,
            stock_name=stock_name,
            dataset_name="financial_statement",
            start_date=start_date,
            end_date=end_date,
            output_path=financial_path,
            reason="source=yfinance; financial_statement not fetched",
        )
        append_not_fetched_log(
            crawler_logs=crawler_logs,
            stock_id=stock_id,
            stock_name=stock_name,
            dataset_name="balance_sheet",
            start_date=start_date,
            end_date=end_date,
            output_path=balance_path,
            reason="source=yfinance; balance_sheet not fetched",
        )

    print(
        "[Info] Rows: "
        f"price={len(price_df)}, "
        f"per={len(per_df)}, "
        f"revenue={len(revenue_df)}, "
        f"financial={len(financial_df)}, "
        f"balance={len(balance_df)}, "
        f"budget={request_budget.status()}"
    )

    return {
        "price": price_df,
        "per": per_df,
        "revenue": revenue_df,
        "financial": financial_df,
        "balance": balance_df,
    }


# =========================
# Load cached raw data for dataset building
# =========================

def load_cached_raw_data_for_tickers(
    tickers_df: pd.DataFrame,
) -> dict[str, list[pd.DataFrame]]:
    """
    從 raw cache 重新載入所有已存在資料。

    這樣即使本次執行中途因 FinMind 402 停止，
    仍可用先前已抓好的 raw CSV 建立 external dataset。
    """
    data = {
        "price": [],
        "per": [],
        "revenue": [],
        "financial": [],
        "balance": [],
    }

    for _, row in tickers_df.iterrows():
        stock_id = str(row["stock_id"])

        paths = {
            "price": EXTERNAL_RAW_PRICE_DIR / f"{stock_id}_price.csv",
            "per": EXTERNAL_RAW_PER_DIR / f"{stock_id}_per_pbr.csv",
            "revenue": EXTERNAL_RAW_REVENUE_DIR / f"{stock_id}_revenue.csv",
            "financial": EXTERNAL_RAW_FINANCIAL_DIR / f"{stock_id}_financial.csv",
            "balance": EXTERNAL_RAW_BALANCE_DIR / f"{stock_id}_balance_sheet.csv",
        }

        for key, path in paths.items():
            df = load_cache_if_exists(path)
            if not df.empty:
                df["stock_id"] = stock_id
                data[key].append(df)

    return data


def validate_finmind_fundamental_coverage(crawler_log_df: pd.DataFrame) -> None:
    print("")
    print("========== External Crawler Coverage Check ==========")

    if crawler_log_df.empty:
        print("[Warning] crawler_log_df is empty.")
        print("========== End Coverage Check ==========")
        return

    summary = (
        crawler_log_df.groupby(["source", "dataset", "fetch_status"], dropna=False)
        .agg(
            n_records=("stock_id", "count"),
            total_rows=("n_rows", "sum"),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    print("")
    print("[Info] FinMind fundamental datasets:")
    for dataset in ["per_pbr", "month_revenue", "financial_statement", "balance_sheet"]:
        sub = crawler_log_df[
            (crawler_log_df["source"] == "finmind")
            & (crawler_log_df["dataset"] == dataset)
        ]

        if sub.empty:
            print(f"[Warning] dataset={dataset}: no FinMind log rows.")
            continue

        print(
            f"- {dataset}: "
            f"success_rate={sub['success'].mean():.2%}, "
            f"total_rows={int(sub['n_rows'].sum())}"
        )

    print("========== End Coverage Check ==========")


# =========================
# Build external dataset
# =========================

def build_and_save_external_dataset(
    *,
    all_prices: list[pd.DataFrame],
    all_per: list[pd.DataFrame],
    all_revenue: list[pd.DataFrame],
    all_financials: list[pd.DataFrame],
    all_balance_sheets: list[pd.DataFrame],
    ticker_name_map: dict[str, str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    price_all_df = (
        pd.concat(all_prices, ignore_index=True)
        if all_prices
        else pd.DataFrame()
    )

    per_all_df = (
        pd.concat(all_per, ignore_index=True)
        if all_per
        else pd.DataFrame()
    )

    revenue_all_df = (
        pd.concat(all_revenue, ignore_index=True)
        if all_revenue
        else pd.DataFrame()
    )

    financial_all_df = (
        pd.concat(all_financials, ignore_index=True)
        if all_financials
        else pd.DataFrame()
    )

    balance_all_df = (
        pd.concat(all_balance_sheets, ignore_index=True)
        if all_balance_sheets
        else pd.DataFrame()
    )

    if price_all_df.empty:
        raise ValueError(
            "沒有任何價格資料，無法建立 Return。"
            "請先確認 data/external/raw/prices 是否已有有效 price csv。"
        )

    print("")
    print("========== Building external dataset ==========")
    print(f"[Info] price rows: {len(price_all_df)}")
    print(f"[Info] per/pbr rows: {len(per_all_df)}")
    print(f"[Info] revenue rows: {len(revenue_all_df)}")
    print(f"[Info] financial statement rows: {len(financial_all_df)}")
    print(f"[Info] balance sheet rows: {len(balance_all_df)}")

    anchor_df = build_anchor_prices(
        price_df=price_all_df,
        start_year=start_year,
        end_year=end_year,
    )

    if anchor_df.empty:
        raise ValueError(
            "無法建立 anchor price。"
            "請檢查 price 資料是否包含 date / close 欄位。"
        )

    returns_df = build_returns_from_anchor_prices(
        anchor_df=anchor_df,
        ticker_name_map=ticker_name_map,
        start_year=start_year,
        end_year=end_year + 1,
    )

    if returns_df.empty:
        raise ValueError(
            "無法建立 Return。"
            "請確認每檔股票至少有 year 與 year+1 的 12 月價格。"
        )

    per_pbr_features = extract_year_end_per_pbr(
        per_df=per_all_df,
        start_year=start_year,
        end_year=end_year,
    )

    revenue_features = extract_revenue_growth(
        revenue_df=revenue_all_df,
        start_year=start_year,
        end_year=end_year,
    )

    financial_features = extract_financial_statement_features(
        financial_df=financial_all_df,
        start_year=start_year,
        end_year=end_year,
    )

    income_items_df = extract_income_statement_items(financial_all_df)
    balance_items_df = extract_balance_sheet_items(balance_all_df)

    engineered_financial_features = calculate_external_financial_features(
        income_items_df=income_items_df,
        balance_items_df=balance_items_df,
        price_return_df=returns_df,
    )

    save_dataframe(
        income_items_df,
        EXTERNAL_RAW_ENGINEERED_DIR / "income_items.csv",
    )
    save_dataframe(
        balance_items_df,
        EXTERNAL_RAW_ENGINEERED_DIR / "balance_items.csv",
    )
    save_dataframe(
        engineered_financial_features,
        EXTERNAL_RAW_ENGINEERED_DIR / "engineered_financial_features.csv",
    )

    print("")
    print("[Info] engineered financial features availability:")
    if engineered_financial_features.empty:
        print("[Warning] engineered_financial_features is empty.")
    else:
        availability = engineered_financial_features.notna().mean()
        print(availability.to_string())

    external_df = merge_external_features(
        returns_df=returns_df,
        per_pbr_features=per_pbr_features,
        revenue_features=revenue_features,
        financial_features=financial_features,
        engineered_financial_features=engineered_financial_features,
    )

    if external_df.empty:
        raise ValueError("合併後 external_df 是空的。請檢查各資料來源與年份。")

    save_dataframe(external_df, EXTERNAL_CLEANED_DATA_PATH)

    profile_text = generate_external_profile(external_df)

    extra_profile_lines = []
    extra_profile_lines.append("")
    extra_profile_lines.append("========== External Financial Feature Engineering ==========")
    extra_profile_lines.append(f"Income item rows: {len(income_items_df)}")
    extra_profile_lines.append(f"Balance item rows: {len(balance_items_df)}")
    extra_profile_lines.append(
        f"Engineered feature rows: {len(engineered_financial_features)}"
    )

    if not engineered_financial_features.empty:
        extra_profile_lines.append("")
        extra_profile_lines.append("Engineered feature availability ratio:")

        excluded_cols = {STOCK_ID_COL, YEAR_COL, "year"}
        feature_cols = [
            col
            for col in engineered_financial_features.columns
            if col not in excluded_cols
        ]

        if feature_cols:
            extra_profile_lines.append(
                engineered_financial_features[feature_cols].notna().mean().to_string()
            )

    extra_profile_lines.append("========== End Financial Feature Engineering ==========")

    full_profile_text = profile_text + "\n" + "\n".join(extra_profile_lines)

    save_text(full_profile_text, EXTERNAL_PROFILE_PATH)

    return external_df


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: External crawler and external dataset builder."
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_EXTERNAL_START_YEAR,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_EXTERNAL_END_YEAR,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="finmind",
        choices=["finmind", "yfinance"],
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional FinMind token. "
            "For safety, prefer FINMIND_TOKEN environment variable instead."
        ),
    )

    parser.add_argument(
        "--max-new-requests",
        type=int,
        default=80,
        help=(
            "Maximum number of new FinMind requests in this run. "
            "Cache hits do not count."
        ),
    )

    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        default=True,
        help="Use existing raw CSV files and avoid repeated requests.",
    )

    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable cache. Not recommended.",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and re-fetch all datasets. Not recommended.",
    )

    parser.add_argument(
        "--stop-on-limit",
        dest="stop_on_limit",
        action="store_true",
        default=True,
        help="Stop the crawler once FinMind request limit is reached.",
    )

    parser.add_argument(
        "--continue-on-limit",
        dest="stop_on_limit",
        action="store_false",
        help="Do not stop on FinMind limit; continue with cache/yfinance where possible.",
    )

    parser.add_argument(
        "--verbose-api",
        action="store_true",
        help="Print FinMind request diagnostics.",
    )

    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional limit for number of tickers to process in this run.",
    )

    args = parser.parse_args()

    ensure_external_dirs()

    start_year = int(args.start_year)
    end_year = int(args.end_year)

    if end_year <= start_year:
        raise ValueError("--end-year 必須大於 --start-year。")

    resolved_token = resolve_finmind_token(args.token)

    if args.source == "finmind":
        if not resolved_token:
            raise ValueError(
                "使用 --source finmind 時必須設定 FINMIND_TOKEN 或傳入 --token。"
            )

        if not is_probably_valid_jwt(resolved_token):
            raise ValueError(
                "FINMIND_TOKEN 格式看起來不正確。"
                "請重新從 FinMind 後台複製完整 token。"
            )

    request_budget = RequestBudget(max_new_requests=args.max_new_requests)

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year + 1}-12-31"

    tickers_df = read_tickers(
        EXTERNAL_TICKERS_PATH,
        max_tickers=args.max_tickers,
    )

    ticker_name_map = dict(zip(tickers_df["stock_id"], tickers_df["stock_name"]))

    crawler_logs: list[dict] = []

    print("========== Step 7: External Crawler ==========")
    print(f"[Info] Source: {args.source}")
    print(f"[Info] Feature years: {start_year} - {end_year}")
    print(f"[Info] Fetch date range: {start_date} - {end_date}")
    print(f"[Info] Number of tickers: {len(tickers_df)}")
    print(f"[Info] Use cache: {args.use_cache}")
    print(f"[Info] Force refresh: {args.force_refresh}")
    print(f"[Info] Max new FinMind requests: {args.max_new_requests}")
    print("")

    for _, row in tickers_df.iterrows():
        stock_id = str(row["stock_id"])
        stock_name = str(row["stock_name"])

        try:
            fetch_one_stock(
                stock_id=stock_id,
                stock_name=stock_name,
                source=args.source,
                start_date=start_date,
                end_date=end_date,
                token=resolved_token,
                crawler_logs=crawler_logs,
                request_budget=request_budget,
                use_cache=args.use_cache,
                force_refresh=args.force_refresh,
                stop_on_limit=args.stop_on_limit,
                verbose_api=args.verbose_api,
            )

        except FinMindRateLimitError as exc:
            print("")
            print("========== FinMind Rate Limit Reached ==========")
            print(str(exc))
            print("[Info] Crawler will stop now and save current logs.")
            print("[Info] Re-run later with --use-cache to continue from existing raw files.")
            print("========== Stop ==========")
            break

    crawler_log_df = pd.DataFrame(crawler_logs)

    if not crawler_log_df.empty:
        save_dataframe(crawler_log_df, EXTERNAL_CRAWLER_LOG_PATH)
        print(f"\n[Saved] {EXTERNAL_CRAWLER_LOG_PATH}")
        validate_finmind_fundamental_coverage(crawler_log_df)
    else:
        print("[Warning] No crawler log was generated in this run.")

    # 用所有已存在 raw cache 建立 dataset，不只用本次成功抓到的資料。
    cached_data = load_cached_raw_data_for_tickers(tickers_df)

    print("")
    print("========== Cached Raw Data Summary ==========")
    print(f"[Info] cached price files: {len(cached_data['price'])}")
    print(f"[Info] cached per/pbr files: {len(cached_data['per'])}")
    print(f"[Info] cached revenue files: {len(cached_data['revenue'])}")
    print(f"[Info] cached financial files: {len(cached_data['financial'])}")
    print(f"[Info] cached balance files: {len(cached_data['balance'])}")
    print("========== End Cached Raw Data Summary ==========")

    external_df = build_and_save_external_dataset(
        all_prices=cached_data["price"],
        all_per=cached_data["per"],
        all_revenue=cached_data["revenue"],
        all_financials=cached_data["financial"],
        all_balance_sheets=cached_data["balance"],
        ticker_name_map=ticker_name_map,
        start_year=start_year,
        end_year=end_year,
    )

    print("")
    print(generate_external_profile(external_df))
    print("")
    print("[Saved outputs]")
    print(f"- cleaned external dataset: {EXTERNAL_CLEANED_DATA_PATH}")
    print(f"- external profile: {EXTERNAL_PROFILE_PATH}")
    print(f"- crawler log: {EXTERNAL_CRAWLER_LOG_PATH}")
    print(f"- engineered income items: {EXTERNAL_RAW_ENGINEERED_DIR / 'income_items.csv'}")
    print(f"- engineered balance items: {EXTERNAL_RAW_ENGINEERED_DIR / 'balance_items.csv'}")
    print(
        f"- engineered financial features: "
        f"{EXTERNAL_RAW_ENGINEERED_DIR / 'engineered_financial_features.csv'}"
    )
    print("")
    print("========== External crawler finished ==========")


if __name__ == "__main__":
    main()