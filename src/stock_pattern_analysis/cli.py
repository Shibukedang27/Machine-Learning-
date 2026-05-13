"""Command-line tools for the stock pattern analysis project."""

from __future__ import annotations

import argparse

from .data_tools import generate_demo_stock
from .model import analyze, load_model, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock pattern analysis with a small ML model")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("generate-demo", help="Create a demo stock CSV")
    demo.add_argument("--output", default="data/stocks/demo_stock.csv")
    demo.add_argument("--days", type=int, default=150)

    train_command = commands.add_parser("train", help="Train the pattern model")
    train_command.add_argument("--csv", default="data/stocks/demo_stock.csv")
    train_command.add_argument("--model", default="models/stock_pattern_model.json")

    analyze_command = commands.add_parser("analyze", help="Analyze the latest pattern")
    analyze_command.add_argument("--csv", default="data/stocks/demo_stock.csv")
    analyze_command.add_argument("--model", default="models/stock_pattern_model.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate-demo":
        generate_demo_stock(args.output, args.days)
        print(f"Demo stock CSV written to {args.output}")
        return 0

    if args.command == "train":
        model = train(args.csv, args.model)
        print(f"Model saved to {args.model}")
        print(f"Training rows: {model['training_rows']}")
        print(f"Training accuracy: {model['training_accuracy']:.2%}")
        return 0

    if args.command == "analyze":
        result = analyze(args.csv, load_model(args.model))
        print(f"Date: {result.date}")
        print(f"Close: {result.close:.2f}")
        print(f"Pattern: {result.pattern}")
        print(f"Probability up next bar: {result.probability_up:.2%}")
        print(f"Confidence: {result.confidence}")
        for note in result.notes:
            print(f"- {note}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
