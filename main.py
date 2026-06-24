# main.py
from __future__ import annotations

import argparse
import subprocess
import sys


COMMANDS = {
    "external-crawl": [
        sys.executable,
        "run_external_crawler.py",
    ],
    "external-rerun": [
        sys.executable,
        "rerun_external_pipeline.py",
    ],
    "svr-portfolio": [
        sys.executable,
        "build_svr_ga_portfolios.py",
    ],
    "rebuild-all-models": [
        sys.executable,
        "rebuild_all_models_portfolio_metrics.py",
    ],
    "web-demo": [
        sys.executable,
        "web_demo.py",
    ],
}


def run_command(command: list[str]) -> None:
    print(f"[Run] {' '.join(command)}")
    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified project entry point."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "external-crawl",
        help="Run external crawler and build external dataset.",
    )

    subparsers.add_parser(
        "external-rerun",
        help="Re-run model and portfolio on external dataset.",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run demo pipeline with a new testing dataset.",
    )
    demo_parser.add_argument("--input", required=True)
    demo_parser.add_argument("--model", default="random_forest")
    demo_parser.add_argument("--top-k", type=int, default=10)

    web_demo_parser = subparsers.add_parser(
        "web-demo",
        help="Run the interactive web demo.",
    )
    web_demo_parser.add_argument("--host", default="127.0.0.1")
    web_demo_parser.add_argument("--port", type=int, default=8000)

    args, unknown_args = parser.parse_known_args()

    if args.command == "demo":
        command = [
            sys.executable,
            "demo.py",
            "--input",
            args.input,
            "--model",
            args.model,
            "--top-k",
            str(args.top_k),
        ]
        run_command(command)
        return

    if args.command in COMMANDS:
        command = COMMANDS[args.command] + unknown_args
        run_command(command)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
