"""Nearest-profile model for WiFi motion identification."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .signal_data import FEATURE_NAMES, WindowFeatures, build_windows, read_wifi_csv, summarize_room


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    distance: float
    room_state: str
    motion_score: float


def train(csv_path: str | Path, model_path: str | Path) -> dict:
    samples = read_wifi_csv(csv_path)
    windows = build_windows(samples)
    if not windows:
        raise ValueError("no feature windows were created")

    means, scales = _fit_scaler([window.features for window in windows])
    grouped: dict[str, list[list[float]]] = {}
    for window in windows:
        grouped.setdefault(window.label, []).append(_scale(window.features, means, scales))

    profiles = {label: _centroid(rows) for label, rows in grouped.items()}
    model = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "means": means,
        "scales": scales,
        "profiles": profiles,
        "training_windows": len(windows),
        "labels": sorted(profiles),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def load_model(model_path: str | Path) -> dict:
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("model feature list does not match this project version")
    return model


def predict(csv_path: str | Path, model: dict) -> Prediction:
    samples = read_wifi_csv(csv_path)
    windows = build_windows(samples)
    latest = windows[-1]
    scaled = _scale(latest.features, model["means"], model["scales"])
    label, distance = _nearest_profile(scaled, model["profiles"])
    confidence = 1.0 / (1.0 + distance)
    room = summarize_room(samples)
    if room["state"] == "empty":
        label = "empty"
    return Prediction(
        label=label,
        confidence=confidence,
        distance=distance,
        room_state=str(room["state"]),
        motion_score=float(room["motion_score"]),
    )


def training_accuracy(csv_path: str | Path, model: dict) -> float:
    windows = build_windows(read_wifi_csv(csv_path))
    correct = 0
    for window in windows:
        scaled = _scale(window.features, model["means"], model["scales"])
        label, _distance = _nearest_profile(scaled, model["profiles"])
        correct += int(label == window.label)
    return correct / len(windows)


def explain_prediction(prediction: Prediction) -> list[str]:
    notes = [
        f"Room state: {prediction.room_state}.",
        f"Motion score: {prediction.motion_score:.3f}.",
        f"Closest trained profile: {prediction.label}.",
    ]
    if prediction.label != "empty":
        notes.append("The label is a consent-based demo profile, not a real identity claim.")
    if prediction.confidence < 0.55:
        notes.append("Confidence is low, so the room movement does not strongly match one profile.")
    return notes


def _fit_scaler(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    columns = list(zip(*rows))
    means = [sum(column) / len(column) for column in columns]
    scales = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _scale(values: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(values, means, scales)]


def _centroid(rows: list[list[float]]) -> list[float]:
    return [sum(column) / len(column) for column in zip(*rows)]


def _nearest_profile(features: list[float], profiles: dict[str, list[float]]) -> tuple[str, float]:
    best_label = ""
    best_distance = float("inf")
    for label, profile in profiles.items():
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(features, profile)))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label, best_distance
