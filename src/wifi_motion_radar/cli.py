"""Command-line tools for the WiFi motion radar project."""

from __future__ import annotations

import argparse

from .model import explain_prediction, load_model, predict, train, training_accuracy
from .signal_data import generate_demo_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WiFi motion radar learning project")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("generate-demo", help="Create synthetic WiFi CSI-like data")
    demo.add_argument("--output", default="data/wifi/demo_wifi_signals.csv")
    demo.add_argument("--samples", type=int, default=180)

    train_command = commands.add_parser("train", help="Train profile centroids")
    train_command.add_argument("--csv", default="data/wifi/demo_wifi_signals.csv")
    train_command.add_argument("--model", default="models/wifi_motion_model.json")

    predict_command = commands.add_parser("predict", help="Predict the latest room movement profile")
    predict_command.add_argument("--csv", default="data/wifi/demo_wifi_signals.csv")
    predict_command.add_argument("--model", default="models/wifi_motion_model.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate-demo":
        generate_demo_dataset(args.output, args.samples)
        print(f"Demo WiFi signal CSV written to {args.output}")
        return 0

    if args.command == "train":
        model = train(args.csv, args.model)
        accuracy = training_accuracy(args.csv, model)
        print(f"Model saved to {args.model}")
        print(f"Training windows: {model['training_windows']}")
        print(f"Training accuracy: {accuracy:.2%}")
        print(f"Profiles: {', '.join(model['labels'])}")
        return 0

    if args.command == "predict":
        prediction = predict(args.csv, load_model(args.model))
        print(f"Room state: {prediction.room_state}")
        print(f"Profile: {prediction.label}")
        print(f"Confidence: {prediction.confidence:.2%}")
        for note in explain_prediction(prediction):
            print(f"- {note}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
