# demo.py
from __future__ import annotations

import argparse

from src.demo_runner import SUPPORTED_DEMO_MODELS, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo-ready stock-selection pipeline."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to demo testing dataset, xlsx or csv.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=SUPPORTED_DEMO_MODELS,
        help="Classification model used for demo prediction.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of selected stocks.",
    )

    args = parser.parse_args()

    run_demo(
        input_path=args.input,
        model_name=args.model,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
