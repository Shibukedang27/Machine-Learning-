"""Command-line interface for the face recognition system."""

from __future__ import annotations

import argparse

from .demo_data import generate_demo_dataset
from .recognizer import load_model, recognize, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local face recognition system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("generate-demo", help="Create synthetic demo faces")
    demo_parser.add_argument("--output", default="data/people")
    demo_parser.add_argument("--samples", type=int, default=4)

    train_parser = subparsers.add_parser("train", help="Train a model from labeled folders")
    train_parser.add_argument("--dataset", default="data/people")
    train_parser.add_argument("--model", default="models/face_model.json")

    recognize_parser = subparsers.add_parser("recognize", help="Recognize one image")
    recognize_parser.add_argument("--model", default="models/face_model.json")
    recognize_parser.add_argument("--image", required=True)
    recognize_parser.add_argument("--threshold", type=float, default=0.72)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate-demo":
        generate_demo_dataset(args.output, args.samples)
        print(f"Demo dataset written to {args.output}")
        return 0

    if args.command == "train":
        model = train(args.dataset, args.model)
        people = ", ".join(sorted(model["profiles"]))
        print(f"Model trained for: {people}")
        print(f"Saved model to {args.model}")
        return 0

    if args.command == "recognize":
        prediction = recognize(load_model(args.model), args.image, args.threshold)
        print(f"Prediction: {prediction.label}")
        print(f"Confidence: {prediction.confidence:.2%}")
        print(f"Similarity score: {prediction.score:.4f}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
