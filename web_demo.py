from __future__ import annotations

import argparse

from src.web_demo import serve_web_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AIFT project web demo.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    serve_web_demo(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
