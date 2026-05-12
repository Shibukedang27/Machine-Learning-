"""Synthetic face-like image generator for tests and demos."""

from __future__ import annotations

import math
from pathlib import Path

from .image_io import GrayImage, write_pgm


PEOPLE = {
    "Ada": {"eye_gap": 13, "mouth_curve": 0.75, "nose_shift": -2},
    "Ben": {"eye_gap": 18, "mouth_curve": -0.35, "nose_shift": 3},
}


def generate_demo_dataset(output_dir: str | Path, samples_per_person: int = 4) -> None:
    output_dir = Path(output_dir)
    for person, traits in PEOPLE.items():
        for sample in range(samples_per_person):
            image = _make_face(
                eye_gap=traits["eye_gap"] + sample % 2,
                mouth_curve=traits["mouth_curve"],
                nose_shift=traits["nose_shift"],
                brightness_shift=(sample - 1) * 5,
            )
            file_name = f"{person.lower()}_{sample + 1:02d}.pgm"
            write_pgm(output_dir / person / file_name, image)


def _make_face(eye_gap: int, mouth_curve: float, nose_shift: int, brightness_shift: int) -> GrayImage:
    width = 64
    height = 64
    pixels: list[list[int]] = []
    for y in range(height):
        row = []
        for x in range(width):
            face_mask = ((x - 32) / 23) ** 2 + ((y - 33) / 27) ** 2 <= 1
            value = 225 if face_mask else 38

            if _ellipse(x, y, 32 - eye_gap // 2, 25, 4, 3) or _ellipse(x, y, 32 + eye_gap // 2, 25, 4, 3):
                value = 35
            if abs(x - (32 + nose_shift)) <= 1 and 30 <= y <= 42:
                value = 115

            mouth_y = 48 + int(math.sin((x - 24) / 16 * math.pi) * mouth_curve * 4)
            if 23 <= x <= 41 and abs(y - mouth_y) <= 1:
                value = 55

            row.append(max(0, min(255, value + brightness_shift)))
        pixels.append(row)
    return GrayImage(width=width, height=height, pixels=pixels)


def _ellipse(x: int, y: int, cx: int, cy: int, rx: int, ry: int) -> bool:
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
