# run_webapp.py
"""
啟動股票篩選 Web Demo。

用法：
    python run_webapp.py                 # http://127.0.0.1:8000
    python run_webapp.py --port 8888
    python run_webapp.py --host 0.0.0.0  # 對外開放（同網段可連）
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stock-selection web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="開發模式自動重載。")
    args = parser.parse_args()

    print(f"[Web Demo] http://{args.host}:{args.port}")
    uvicorn.run(
        "webapp.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
