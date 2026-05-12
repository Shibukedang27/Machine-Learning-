"""Lightweight face embedding and profile matching."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .image_io import GrayImage, read_image

EMBED_WIDTH = 24
EMBED_HEIGHT = 24


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    score: float


def resize_to_embedding(image: GrayImage) -> list[float]:
    resized: list[float] = []
    for y in range(EMBED_HEIGHT):
        source_y = min(image.height - 1, round(y * (image.height - 1) / max(1, EMBED_HEIGHT - 1)))
        for x in range(EMBED_WIDTH):
            source_x = min(image.width - 1, round(x * (image.width - 1) / max(1, EMBED_WIDTH - 1)))
            resized.append(image.pixels[source_y][source_x] / 255.0)
    return _normalize(resized)


def _normalize(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    magnitude = math.sqrt(sum(value * value for value in centered))
    if magnitude == 0:
        return centered
    return [value / magnitude for value in centered]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embeddings must have the same length")
    return sum(a * b for a, b in zip(left, right))


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("at least one embedding is required")
    length = len(embeddings[0])
    totals = [0.0] * length
    for embedding in embeddings:
        if len(embedding) != length:
            raise ValueError("all embeddings must have the same length")
        for index, value in enumerate(embedding):
            totals[index] += value
    return _normalize([total / len(embeddings) for total in totals])


def train(dataset_dir: str | Path, model_path: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    profiles: dict[str, list[float]] = {}
    samples: dict[str, int] = {}

    for person_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        embeddings = []
        for image_path in sorted(person_dir.glob("*")):
            if image_path.suffix.lower() not in {".pgm", ".ppm"}:
                continue
            embeddings.append(resize_to_embedding(read_image(image_path)))
        if embeddings:
            profiles[person_dir.name] = average_embeddings(embeddings)
            samples[person_dir.name] = len(embeddings)

    if not profiles:
        raise ValueError(f"no training images found in {dataset_dir}")

    model = {
        "version": 1,
        "embedding_size": EMBED_WIDTH * EMBED_HEIGHT,
        "profiles": profiles,
        "samples": samples,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def load_model(model_path: str | Path) -> dict:
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if "profiles" not in model:
        raise ValueError("model is missing profiles")
    return model


def recognize(model: dict, image_path: str | Path, threshold: float = 0.72) -> Prediction:
    embedding = resize_to_embedding(read_image(image_path))
    best_label = "Unknown"
    best_score = -1.0

    for label, profile in model["profiles"].items():
        score = cosine_similarity(embedding, profile)
        if score > best_score:
            best_label = label
            best_score = score

    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    if best_score < threshold:
        best_label = "Unknown"
    return Prediction(label=best_label, confidence=confidence, score=best_score)
