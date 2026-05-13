"""Synthetic WiFi signal data and feature extraction."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


SUBCARRIERS = 16
FEATURE_NAMES = [
    "avg_motion",
    "peak_motion",
    "signal_energy",
    "breathing_band",
    "walking_band",
    "left_right_balance",
    "profile_shape",
]


@dataclass(frozen=True)
class WifiSample:
    timestamp: float
    label: str
    values: list[float]


@dataclass(frozen=True)
class WindowFeatures:
    label: str
    start_time: float
    end_time: float
    features: list[float]


PROFILES = {
    "empty": {"motion": 0.02, "pace": 0.0, "shape": 0.0, "side": 0.0},
    "Aarav_walking": {"motion": 0.42, "pace": 1.15, "shape": 0.28, "side": -0.18},
    "Maya_walking": {"motion": 0.34, "pace": 0.82, "shape": -0.22, "side": 0.24},
    "visitor_moving": {"motion": 0.38, "pace": 1.45, "shape": 0.08, "side": 0.04},
}


def generate_demo_dataset(path: str | Path, samples_per_label: int = 180) -> None:
    rows: list[WifiSample] = []
    timestamp = 0.0
    for label, params in PROFILES.items():
        for index in range(samples_per_label):
            rows.append(_make_sample(timestamp, label, index, params))
            timestamp += 0.1
    write_wifi_csv(path, rows)


def read_wifi_csv(path: str | Path) -> list[WifiSample]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"timestamp", "label"}.union({f"subcarrier_{index + 1}" for index in range(SUBCARRIERS)})
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        samples = [
            WifiSample(
                timestamp=float(row["timestamp"]),
                label=row["label"],
                values=[float(row[f"subcarrier_{index + 1}"]) for index in range(SUBCARRIERS)],
            )
            for row in reader
        ]
    if len(samples) < 40:
        raise ValueError("at least 40 WiFi samples are needed")
    return samples


def write_wifi_csv(path: str | Path, rows: list[WifiSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "label", *[f"subcarrier_{index + 1}" for index in range(SUBCARRIERS)]])
        for row in rows:
            writer.writerow([f"{row.timestamp:.1f}", row.label, *[f"{value:.5f}" for value in row.values]])


def build_windows(samples: list[WifiSample], window_size: int = 30, step: int = 15) -> list[WindowFeatures]:
    windows: list[WindowFeatures] = []
    for start in range(0, len(samples) - window_size + 1, step):
        chunk = samples[start : start + window_size]
        label = _majority_label(chunk)
        windows.append(
            WindowFeatures(
                label=label,
                start_time=chunk[0].timestamp,
                end_time=chunk[-1].timestamp,
                features=extract_features(chunk),
            )
        )
    return windows


def extract_features(samples: list[WifiSample]) -> list[float]:
    changes = []
    for previous, current in zip(samples, samples[1:]):
        changes.append(sum(abs(a - b) for a, b in zip(previous.values, current.values)) / SUBCARRIERS)

    avg_motion = sum(changes) / len(changes)
    peak_motion = max(changes)
    all_values = [value for sample in samples for value in sample.values]
    mean_value = sum(all_values) / len(all_values)
    signal_energy = sum((value - mean_value) ** 2 for value in all_values) / len(all_values)

    series = [sum(sample.values) / SUBCARRIERS for sample in samples]
    breathing_band = _band_energy(series, 1)
    walking_band = _band_energy(series, 4)
    left = sum(sum(sample.values[: SUBCARRIERS // 2]) for sample in samples)
    right = sum(sum(sample.values[SUBCARRIERS // 2 :]) for sample in samples)
    left_right_balance = (left - right) / (abs(left) + abs(right) + 1e-9)
    profile_shape = sum((index - 7.5) * value for sample in samples for index, value in enumerate(sample.values)) / len(samples)

    return [
        avg_motion,
        peak_motion,
        signal_energy,
        breathing_band,
        walking_band,
        left_right_balance,
        profile_shape / 100.0,
    ]


def summarize_room(samples: list[WifiSample]) -> dict[str, float | str]:
    latest = samples[-30:]
    motion = extract_features(latest)[0]
    state = "empty" if motion < 0.05 else "movement detected"
    return {"state": state, "motion_score": motion, "samples": float(len(samples))}


def _make_sample(timestamp: float, label: str, index: int, params: dict[str, float]) -> WifiSample:
    values = []
    motion = params["motion"]
    pace = params["pace"]
    shape = params["shape"]
    side = params["side"]
    for carrier in range(SUBCARRIERS):
        carrier_center = (carrier - 7.5) / 7.5
        base = 52.0 + carrier_center * shape * 8.0
        walking_wave = math.sin(index * pace * 0.38 + carrier * 0.55) * motion * 7.0
        breathing_wave = math.sin(index * 0.11 + carrier * 0.2) * (0.25 if label != "empty" else 0.05)
        side_bias = side * carrier_center * 2.0
        deterministic_noise = math.sin(index * 1.91 + carrier * 2.17) * 0.18
        values.append(base + walking_wave + breathing_wave + side_bias + deterministic_noise)
    return WifiSample(timestamp=timestamp, label=label, values=values)


def _majority_label(samples: list[WifiSample]) -> str:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return max(counts, key=counts.get)


def _band_energy(values: list[float], cycles: int) -> float:
    sine = 0.0
    cosine = 0.0
    for index, value in enumerate(values):
        angle = 2 * math.pi * cycles * index / len(values)
        sine += value * math.sin(angle)
        cosine += value * math.cos(angle)
    return math.sqrt(sine * sine + cosine * cosine) / len(values)
