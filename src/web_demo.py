from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from src.config import CLEANED_DATA_PATH, DEMO_OUTPUT_DIR, PROJECT_ROOT, SAVED_MODEL_DIR
from src.demo_runner import (
    MODEL_METADATA,
    SUPPORTED_DEMO_MODELS,
    prepare_demo_data,
    run_demo_analysis,
)
from src.project_dashboard import (
    build_project_overview_payload,
    dataframe_to_ui_records,
)
from src.schema import STOCK_ID_COL, STOCK_NAME_COL, YEAR_COL


WEB_STATIC_DIR = PROJECT_ROOT / "web"
WEB_UPLOAD_DIR = PROJECT_ROOT / "tmp" / "web_uploads"
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    stored_path: Path
    summary: dict[str, Any]


@dataclass
class AppState:
    uploads: dict[str, UploadRecord] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric = float(value)

        if np.isnan(numeric) or np.isinf(numeric):
            return None

        return round(numeric, 6)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if value is pd.NA or pd.isna(value):
        return None

    return value


def _frame_to_records(
    df: pd.DataFrame,
    *,
    sort_by: list[str] | None = None,
    ascending: list[bool] | None = None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    view_df = df.copy()

    if sort_by:
        view_df = view_df.sort_values(sort_by, ascending=ascending)

    return dataframe_to_ui_records(view_df)


def _saved_model_pattern(model_name: str) -> str:
    if model_name == "decision_tree_entropy":
        return "decision_tree_split_*.joblib"

    return f"{model_name}_split_*.joblib"


def build_health_payload() -> dict[str, Any]:
    saved_models = {}

    for model_name in SUPPORTED_DEMO_MODELS:
        saved_models[model_name] = len(
            list(Path(SAVED_MODEL_DIR).glob(_saved_model_pattern(model_name)))
        )

    return {
        "status": "ok",
        "cleaned_data_exists": Path(CLEANED_DATA_PATH).exists(),
        "saved_model_dir_exists": Path(SAVED_MODEL_DIR).exists(),
        "saved_models": saved_models,
        "demo_output_dir": str(DEMO_OUTPUT_DIR),
        "static_dir": str(WEB_STATIC_DIR),
    }


def build_models_payload() -> dict[str, Any]:
    return {
        "models": [
            {
                "model_name": model_name,
                **MODEL_METADATA[model_name],
            }
            for model_name in SUPPORTED_DEMO_MODELS
        ]
    }


def build_upload_payload(record: UploadRecord) -> dict[str, Any]:
    return {
        "upload_id": record.upload_id,
        "filename": record.filename,
        "stored_path": str(record.stored_path),
        "summary": record.summary,
    }


def build_run_payload(run_id: str, result_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        **result_payload,
    }


def build_demo_result_payload(result) -> dict[str, Any]:
    return {
        "request": {
            "input_path": str(result.input_path),
            "model_name": result.model_name,
            "top_k": result.top_k,
        },
        "input_summary": result.input_summary,
        "evaluation_summary": result.evaluation_summary,
        "model_sources": result.model_sources,
        "artifacts": {key: str(path) for key, path in result.output_paths.items()},
        "tables": {
            "predictions": _frame_to_records(
                result.predictions_df,
                sort_by=[YEAR_COL, "score_label_1", STOCK_ID_COL],
                ascending=[True, False, True],
            ),
            "selected_stocks": _frame_to_records(
                result.selected_with_weights_df,
                sort_by=[YEAR_COL, "rank"],
                ascending=[True, True],
            ),
            "portfolio_returns": _frame_to_records(
                result.portfolio_returns_df,
                sort_by=[YEAR_COL],
                ascending=[True],
            ),
            "portfolio_metrics": _frame_to_records(result.portfolio_metrics_df),
            "classification_metrics": _frame_to_records(result.classification_metrics_df),
        },
        "profile_text": result.profile_text,
    }


class WebDemoRequestHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        directory: str | None = None,
        app_state: AppState,
        **kwargs,
    ) -> None:
        self.app_state = app_state
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"[WebDemo] {format % args}")

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self,
        message: str,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _send_file(self, path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()

        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        if not raw_body:
            return {}

        return json.loads(raw_body.decode("utf-8"))

    def _ensure_upload_size_allowed(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))

        if content_length > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"上傳檔案過大。大小上限為 {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB。"
            )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path.startswith("/artifacts/"):
                self._handle_artifact(parsed.path)
                return

            if parsed.path == "/api/health":
                self._send_json({"ok": True, **build_health_payload()})
                return

            if parsed.path == "/api/models":
                self._send_json({"ok": True, **build_models_payload()})
                return

            if parsed.path == "/api/project/overview":
                self._send_json({"ok": True, **build_project_overview_payload()})
                return

            if parsed.path.startswith("/api/demo/result/"):
                run_id = parsed.path.rsplit("/", 1)[-1]
                payload = self.app_state.results.get(run_id)

                if payload is None:
                    self._send_error_json("找不到執行結果。", status=HTTPStatus.NOT_FOUND)
                    return

                self._send_json({"ok": True, **payload})
                return

            if parsed.path in {"/", ""}:
                self.path = "/index.html"
                return super().do_GET()

            candidate = WEB_STATIC_DIR / parsed.path.lstrip("/")
            if candidate.exists():
                return super().do_GET()

            self.path = "/index.html"
            return super().do_GET()
        except Exception as exc:  # pragma: no cover - defensive web handler
            self._send_error_json(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_artifact(self, raw_path: str) -> None:
        relative_text = unquote(raw_path.replace("/artifacts/", "", 1)).strip("/")
        target = (PROJECT_ROOT / relative_text).resolve()
        project_root = PROJECT_ROOT.resolve()

        if project_root != target and project_root not in target.parents:
            self._send_error_json("Artifact 路徑超出專案根目錄。", status=HTTPStatus.FORBIDDEN)
            return

        if not target.exists() or not target.is_file():
            self._send_error_json("找不到 Artifact 檔案。", status=HTTPStatus.NOT_FOUND)
            return

        self._send_file(target)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/demo/upload":
                self._handle_upload()
                return

            if parsed.path == "/api/demo/run":
                self._handle_run()
                return

            self._send_error_json("未知的 API 端點。", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - defensive web handler
            self._send_error_json(str(exc), status=HTTPStatus.BAD_REQUEST)

    def _handle_upload(self) -> None:
        self._ensure_upload_size_allowed()
        payload = self._read_json_body()

        filename = Path(str(payload.get("filename", ""))).name
        content_base64 = str(payload.get("content_base64", ""))

        if not filename:
            raise ValueError("上傳檔案缺少檔名。")

        if not content_base64:
            raise ValueError("上傳內容為空。")

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
            raise ValueError(f"不支援的檔案類型：{suffix}。允許類型：{allowed}。")

        try:
            file_bytes = base64.b64decode(content_base64)
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise ValueError("上傳內容不是合法的 base64。") from exc

        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"上傳檔案過大。大小上限為 {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB。"
            )

        upload_id = uuid.uuid4().hex[:12]
        stored_path = WEB_UPLOAD_DIR / f"{upload_id}{suffix}"
        WEB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        with stored_path.open("wb") as handle:
            handle.write(file_bytes)

        prepared_df, summary = prepare_demo_data(stored_path)

        record = UploadRecord(
            upload_id=upload_id,
            filename=filename,
            stored_path=stored_path,
            summary=summary,
        )
        self.app_state.uploads[upload_id] = record

        self._send_json(
            {
                "ok": True,
                "upload": build_upload_payload(record),
                "preview_rows": _frame_to_records(prepared_df.head(20)),
            }
        )

    def _handle_run(self) -> None:
        payload = self._read_json_body()

        upload_id = str(payload.get("upload_id", "")).strip()
        model_name = str(payload.get("model_name", "random_forest")).strip()
        top_k = int(payload.get("top_k", 10))

        if not upload_id:
            raise ValueError("缺少必要欄位 upload_id。")

        if model_name not in SUPPORTED_DEMO_MODELS:
            raise ValueError(f"不支援的模型：{model_name}")

        record = self.app_state.uploads.get(upload_id)
        if record is None:
            raise ValueError("找不到上傳紀錄，請先上傳並驗證資料集。")

        result = run_demo_analysis(
            input_path=record.stored_path,
            model_name=model_name,
            top_k=top_k,
            persist_outputs=True,
        )

        run_id = uuid.uuid4().hex[:12]
        result_payload = build_demo_result_payload(result)
        stored_payload = build_run_payload(run_id, result_payload)
        self.app_state.results[run_id] = stored_payload

        self._send_json({"ok": True, **stored_payload})


def serve_web_demo(host: str = "127.0.0.1", port: int = 8000) -> None:
    if not WEB_STATIC_DIR.exists():
        raise FileNotFoundError(f"找不到 Web UI 目錄：{WEB_STATIC_DIR}")

    WEB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path(DEMO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    app_state = AppState()
    handler = partial(
        WebDemoRequestHandler,
        directory=str(WEB_STATIC_DIR),
        app_state=app_state,
    )

    with ThreadingHTTPServer((host, port), handler) as httpd:
        print("========== Web Demo ==========")
        print(f"[Info] Server: http://{host}:{port}")
        print(f"[Info] Static UI: {WEB_STATIC_DIR}")
        print(f"[Info] Upload cache: {WEB_UPLOAD_DIR}")
        print("[Info] Press Ctrl+C to stop the server.")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("")
            print("[Info] Web demo server stopped.")
