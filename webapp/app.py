# webapp/app.py
"""
FastAPI app for the stock-selection web demo.

API（對應 README 15.7，MVP 版本）：
- GET  /api/health
- GET  /api/demo/models
- GET  /api/demo/schema
- POST /api/demo/run   (multipart: file, model, top_k)

前端為單頁靜態頁面，掛載於 /。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.service import (
    DemoInputError,
    feature_schema,
    list_models,
    pretrained_status,
    run_inference,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="Stock Selection Web Demo", version="1.0.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo/models")
def get_models() -> dict[str, object]:
    return {"models": list_models()}


@app.get("/api/demo/schema")
def get_schema() -> dict[str, object]:
    return feature_schema()


@app.get("/api/demo/pretrained")
def get_pretrained() -> dict[str, object]:
    return pretrained_status()


@app.post("/api/demo/run")
async def run_demo_endpoint(
    file: UploadFile = File(...),
    model: str = Form("random_forest"),
    top_k: int = Form(10),
    strategy: str = Form("retrain"),
) -> JSONResponse:
    filename = file.filename or "uploaded"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": (
                    f"不支援的檔案格式「{suffix or '未知'}」。"
                    "請上傳 .csv / .xlsx / .xls。"
                ),
            },
        )

    content = await file.read()
    if not content:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "上傳的檔案是空的。"},
        )
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "檔案過大（上限 20 MB）。"},
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        result = run_inference(
            input_path=tmp_path,
            model_name=model,
            top_k=int(top_k),
            strategy=strategy,
        )
        return JSONResponse(content=result)

    except DemoInputError as exc:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001 - 回傳通用錯誤，避免洩漏 traceback
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"伺服器推論失敗：{exc}"},
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# 靜態資源（CSS/JS 若拆檔可放這裡）。放在最後，避免蓋掉 /api 路由。
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
