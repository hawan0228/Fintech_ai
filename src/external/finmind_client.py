# src/external/finmind_client.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindAPIError(RuntimeError):
    """General FinMind API error."""


class FinMindRateLimitError(FinMindAPIError):
    """Raised when FinMind returns request upper-limit error."""


def resolve_finmind_token(token: str | None = None) -> str | None:
    if token and str(token).strip():
        return str(token).strip()

    env_token = os.getenv("FINMIND_TOKEN")

    if env_token and env_token.strip():
        return env_token.strip()

    return None


def mask_token(text: str, token: str | None) -> str:
    if not text or not token:
        return text

    return text.replace(token, "***TOKEN_MASKED***")


def is_probably_valid_jwt(token: str | None) -> bool:
    if not token:
        return False

    parts = token.split(".")

    if len(parts) != 3:
        return False

    return parts[0].startswith("eyJ")


def extract_payload_message(payload: dict[str, Any]) -> str:
    items = []

    for key in ["status", "msg", "message", "error", "detail"]:
        if key in payload:
            items.append(f"{key}={payload.get(key)}")

    if not items:
        items.append(f"payload_keys={list(payload.keys())}")

    return "; ".join(items)


def parse_response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception as exc:
        raise FinMindAPIError(
            f"Response is not valid JSON. "
            f"HTTP={response.status_code}, text={response.text[:500]}"
        ) from exc

    if not isinstance(payload, dict):
        raise FinMindAPIError(
            f"Response JSON is not a dict. "
            f"HTTP={response.status_code}, type={type(payload)}"
        )

    return payload


def request_finmind_dataset(
    dataset: str,
    data_id: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
    verbose: bool = False,
) -> pd.DataFrame:
    resolved_token = resolve_finmind_token(token)

    if not resolved_token:
        raise FinMindAPIError(
            "FINMIND_TOKEN is not set. Please set environment variable FINMIND_TOKEN."
        )

    if not is_probably_valid_jwt(resolved_token):
        raise FinMindAPIError(
            "FINMIND_TOKEN format looks suspicious. Please copy a complete token."
        )

    params = {
        "dataset": dataset,
        "data_id": str(data_id),
        "start_date": start_date,
        "end_date": end_date,
        "token": resolved_token,
    }

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            safe_params = params.copy()
            safe_params["token"] = "***TOKEN_MASKED***"

            if verbose:
                print(f"[FinMind] attempt={attempt}, params={safe_params}")

            response = requests.get(
                FINMIND_API_URL,
                params=params,
                timeout=timeout,
            )

            safe_url = mask_token(response.url, resolved_token)
            payload = parse_response_json(response)
            message = extract_payload_message(payload)

            if verbose:
                print(f"[FinMind] HTTP status={response.status_code}")
                print(f"[FinMind] URL={safe_url}")
                print(f"[FinMind] message={message}")

            status_from_payload = payload.get("status")

            # FinMind 超額常見為 HTTP 402 / payload status 402。
            if response.status_code == 402 or str(status_from_payload) == "402":
                raise FinMindRateLimitError(
                    f"FinMind request limit reached. "
                    f"dataset={dataset}, data_id={data_id}, "
                    f"message={message}, url={safe_url}"
                )

            # 其他 4xx 不應盲目重試太多次。
            if 400 <= response.status_code < 500:
                raise FinMindAPIError(
                    f"HTTP {response.status_code}. "
                    f"dataset={dataset}, data_id={data_id}, "
                    f"message={message}, url={safe_url}"
                )

            if response.status_code >= 500:
                raise FinMindAPIError(
                    f"HTTP {response.status_code}. "
                    f"dataset={dataset}, data_id={data_id}, "
                    f"message={message}, url={safe_url}"
                )

            if "data" not in payload:
                raise FinMindAPIError(
                    f"Payload has no data field. "
                    f"dataset={dataset}, data_id={data_id}, message={message}"
                )

            data = payload.get("data")

            if data is None:
                raise FinMindAPIError(
                    f"Payload data is None. "
                    f"dataset={dataset}, data_id={data_id}, message={message}"
                )

            if not isinstance(data, list):
                raise FinMindAPIError(
                    f"Payload data is not a list. "
                    f"dataset={dataset}, data_id={data_id}, "
                    f"type={type(data)}, message={message}"
                )

            return pd.DataFrame(data)

        except FinMindRateLimitError:
            # rate limit 不 retry，避免浪費額度與觸發封鎖。
            raise

        except Exception as exc:
            last_error = exc

            if attempt < max_retries:
                wait = sleep_seconds * attempt
                print(
                    f"[FinMind Warning] dataset={dataset}, stock_id={data_id}, "
                    f"attempt={attempt}/{max_retries} failed: {exc}. "
                    f"Retry after {wait:.1f}s."
                )
                time.sleep(wait)
            else:
                break

    raise FinMindAPIError(
        f"FinMind request failed after {max_retries} attempts. "
        f"dataset={dataset}, data_id={data_id}, "
        f"start={start_date}, end={end_date}, "
        f"last_error={last_error}"
    )


def fetch_finmind_stock_price(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep_seconds: float = 0.35,
    verbose: bool = False,
) -> pd.DataFrame:
    df = request_finmind_dataset(
        dataset="TaiwanStockPrice",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
        verbose=verbose,
    )
    time.sleep(sleep_seconds)
    return df


def fetch_finmind_per_pbr(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep_seconds: float = 0.35,
    verbose: bool = False,
) -> pd.DataFrame:
    df = request_finmind_dataset(
        dataset="TaiwanStockPER",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
        verbose=verbose,
    )
    time.sleep(sleep_seconds)
    return df


def fetch_finmind_month_revenue(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep_seconds: float = 0.35,
    verbose: bool = False,
) -> pd.DataFrame:
    df = request_finmind_dataset(
        dataset="TaiwanStockMonthRevenue",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
        verbose=verbose,
    )
    time.sleep(sleep_seconds)
    return df


def fetch_finmind_financial_statement(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep_seconds: float = 0.35,
    verbose: bool = False,
) -> pd.DataFrame:
    df = request_finmind_dataset(
        dataset="TaiwanStockFinancialStatements",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
        verbose=verbose,
    )
    time.sleep(sleep_seconds)
    return df


def fetch_finmind_balance_sheet(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep_seconds: float = 0.35,
    verbose: bool = False,
) -> pd.DataFrame:
    df = request_finmind_dataset(
        dataset="TaiwanStockBalanceSheet",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
        verbose=verbose,
    )
    time.sleep(sleep_seconds)
    return df


def save_raw_df(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")