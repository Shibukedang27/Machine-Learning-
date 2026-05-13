"""CSV loading, demo data, and feature building for stock pattern analysis."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass(frozen=True)
class PriceBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FeatureRow:
    date: str
    close: float
    features: list[float]
    target: int | None


FEATURE_NAMES = [
    "daily_return",
    "five_day_return",
    "ten_day_return",
    "ma_gap",
    "volatility",
    "volume_pressure",
    "range_position",
]


def read_price_csv(path: str | Path) -> list[PriceBar]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        rows = [
            PriceBar(
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reader
        ]

    if len(rows) < 25:
        raise ValueError("at least 25 rows are needed for pattern analysis")
    return rows


def write_price_csv(path: str | Path, rows: list[PriceBar]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for row in rows:
            writer.writerow([row.date, f"{row.open:.2f}", f"{row.high:.2f}", f"{row.low:.2f}", f"{row.close:.2f}", f"{row.volume:.0f}"])


def moving_average(values: list[float], end: int, window: int) -> float:
    start = end - window + 1
    if start < 0:
        raise ValueError("not enough values for the requested moving average")
    return sum(values[start : end + 1]) / window


def build_feature_rows(rows: list[PriceBar]) -> list[FeatureRow]:
    closes = [row.close for row in rows]
    volumes = [row.volume for row in rows]
    feature_rows: list[FeatureRow] = []

    for index in range(20, len(rows)):
        close = closes[index]
        previous_close = closes[index - 1]
        daily_return = _safe_ratio(close - previous_close, previous_close)
        five_day_return = _safe_ratio(close - closes[index - 5], closes[index - 5])
        ten_day_return = _safe_ratio(close - closes[index - 10], closes[index - 10])
        ma_5 = moving_average(closes, index, 5)
        ma_20 = moving_average(closes, index, 20)
        ma_gap = _safe_ratio(ma_5 - ma_20, ma_20)

        recent_returns = [_safe_ratio(closes[i] - closes[i - 1], closes[i - 1]) for i in range(index - 9, index + 1)]
        volatility = _stddev(recent_returns)
        volume_pressure = _safe_ratio(volumes[index] - moving_average(volumes, index, 20), moving_average(volumes, index, 20))
        high_20 = max(row.high for row in rows[index - 19 : index + 1])
        low_20 = min(row.low for row in rows[index - 19 : index + 1])
        range_position = _safe_ratio(close - low_20, high_20 - low_20)

        target = None
        if index < len(rows) - 1:
            target = 1 if closes[index + 1] > close else 0

        feature_rows.append(
            FeatureRow(
                date=rows[index].date,
                close=close,
                features=[
                    daily_return,
                    five_day_return,
                    ten_day_return,
                    ma_gap,
                    volatility,
                    volume_pressure,
                    range_position,
                ],
                target=target,
            )
        )
    return feature_rows


def generate_demo_stock(path: str | Path, days: int = 150) -> None:
    rows: list[PriceBar] = []
    close = 100.0
    volume_base = 1_200_000
    start_date = date(2026, 1, 1)

    for day in range(days):
        wave = math.sin(day / 8) * 0.9
        trend = 0.11 if day < 70 else (-0.04 if day < 105 else 0.16)
        breakout = 1.7 if 108 <= day <= 118 else 0.0
        shock = math.sin(day * 1.7) * 0.35
        open_price = close + math.sin(day / 5) * 0.45
        close = max(12.0, close + trend + wave + shock + breakout)
        high = max(open_price, close) + 0.8 + abs(math.sin(day)) * 0.7
        low = min(open_price, close) - 0.8 - abs(math.cos(day)) * 0.5
        volume = volume_base + day * 2800 + abs(wave) * 130_000 + (380_000 if breakout else 0)
        rows.append(
            PriceBar(
                date=(start_date + timedelta(days=day)).isoformat(),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )

    write_price_csv(path, rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _stddev(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
