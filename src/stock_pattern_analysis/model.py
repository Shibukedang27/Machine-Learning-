"""A compact machine-learning model for stock pattern direction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .data_tools import FEATURE_NAMES, FeatureRow, PriceBar, build_feature_rows, read_price_csv


@dataclass(frozen=True)
class Analysis:
    date: str
    close: float
    pattern: str
    probability_up: float
    confidence: str
    notes: list[str]


def train(csv_path: str | Path, model_path: str | Path, epochs: int = 900, learning_rate: float = 0.08) -> dict:
    rows = read_price_csv(csv_path)
    feature_rows = [row for row in build_feature_rows(rows) if row.target is not None]
    if len(feature_rows) < 20:
        raise ValueError("not enough labeled rows to train the model")

    means, scales = _fit_scaler([row.features for row in feature_rows])
    weights = [0.0] * len(FEATURE_NAMES)
    bias = 0.0

    for _ in range(epochs):
        for row in feature_rows:
            scaled = _scale(row.features, means, scales)
            probability = _sigmoid(_dot(weights, scaled) + bias)
            error = probability - float(row.target)
            for index, value in enumerate(scaled):
                weights[index] -= learning_rate * error * value
            bias -= learning_rate * error

    correct = 0
    for row in feature_rows:
        probability = _sigmoid(_dot(weights, _scale(row.features, means, scales)) + bias)
        correct += int((probability >= 0.5) == bool(row.target))

    model = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "means": means,
        "scales": scales,
        "weights": weights,
        "bias": bias,
        "training_rows": len(feature_rows),
        "training_accuracy": correct / len(feature_rows),
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


def analyze(csv_path: str | Path, model: dict) -> Analysis:
    rows = read_price_csv(csv_path)
    feature_rows = build_feature_rows(rows)
    latest = feature_rows[-1]
    probability_up = predict_probability(model, latest.features)
    pattern = describe_pattern(rows, latest, probability_up)
    confidence = "high" if probability_up >= 0.68 or probability_up <= 0.32 else "medium" if abs(probability_up - 0.5) >= 0.08 else "low"
    return Analysis(
        date=latest.date,
        close=latest.close,
        pattern=pattern,
        probability_up=probability_up,
        confidence=confidence,
        notes=build_notes(rows, latest, probability_up),
    )


def predict_probability(model: dict, features: list[float]) -> float:
    scaled = _scale(features, model["means"], model["scales"])
    return _sigmoid(_dot(model["weights"], scaled) + model["bias"])


def describe_pattern(rows: list[PriceBar], latest: FeatureRow, probability_up: float) -> str:
    daily_return, five_day_return, ten_day_return, ma_gap, volatility, volume_pressure, range_position = latest.features

    if probability_up >= 0.62 and range_position > 0.78 and volume_pressure > 0.12:
        return "bullish breakout setup"
    if probability_up >= 0.58 and ma_gap > 0 and five_day_return < 0:
        return "trend pullback with recovery potential"
    if probability_up <= 0.42 and ma_gap < 0 and ten_day_return < 0:
        return "bearish continuation risk"
    if volatility > 0.025 and abs(daily_return) > 0.018:
        return "volatile decision zone"
    if 0.35 < range_position < 0.65 and abs(ma_gap) < 0.01:
        return "sideways consolidation"
    return "mixed pattern"


def build_notes(rows: list[PriceBar], latest: FeatureRow, probability_up: float) -> list[str]:
    daily_return, five_day_return, ten_day_return, ma_gap, volatility, volume_pressure, range_position = latest.features
    notes = [
        f"Model leans {'upward' if probability_up >= 0.5 else 'downward'} for the next bar.",
        f"Five-day move: {five_day_return * 100:.2f}%.",
        f"Twenty-day range position: {range_position * 100:.1f}%.",
    ]
    if volume_pressure > 0.2:
        notes.append("Volume is meaningfully above its recent average, so the move has more participation.")
    if volatility > 0.03:
        notes.append("Volatility is elevated; treat the signal with extra caution.")
    if ma_gap > 0.02:
        notes.append("Short-term price action is sitting above the longer moving average.")
    elif ma_gap < -0.02:
        notes.append("Short-term price action is below the longer moving average.")
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


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    if value < -50:
        return 0.0
    if value > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))
