"""Small PGM/PPM reader and writer used to keep the project dependency-free."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GrayImage:
    width: int
    height: int
    pixels: list[list[int]]

    def flattened(self) -> list[int]:
        return [value for row in self.pixels for value in row]


def _tokenize_pnm(data: bytes) -> list[bytes]:
    tokens: list[bytes] = []
    i = 0
    while i < len(data):
        byte = data[i]
        if byte in b" \t\r\n":
            i += 1
            continue
        if byte == ord("#"):
            while i < len(data) and data[i] not in b"\r\n":
                i += 1
            continue
        start = i
        while i < len(data) and data[i] not in b" \t\r\n":
            i += 1
        tokens.append(data[start:i])
    return tokens


def read_image(path: str | Path) -> GrayImage:
    path = Path(path)
    data = path.read_bytes()
    if not data:
        raise ValueError(f"{path} is empty")

    magic = data[:2]
    if magic in {b"P2", b"P3"}:
        return _read_ascii_pnm(path, data)
    if magic in {b"P5", b"P6"}:
        return _read_binary_pnm(path, data)
    raise ValueError(f"{path} is not a supported PGM/PPM image")


def _read_ascii_pnm(path: Path, data: bytes) -> GrayImage:
    tokens = _tokenize_pnm(data)
    if len(tokens) < 4:
        raise ValueError(f"{path} has an incomplete PNM header")

    magic = tokens[0]
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if max_value <= 0:
        raise ValueError(f"{path} has an invalid max value")

    expected = width * height * (3 if magic == b"P3" else 1)
    values = [int(token) for token in tokens[4:]]
    if len(values) < expected:
        raise ValueError(f"{path} does not contain enough pixel data")

    if magic == b"P2":
        gray_values = values[: width * height]
    else:
        gray_values = []
        for i in range(0, expected, 3):
            red, green, blue = values[i], values[i + 1], values[i + 2]
            gray_values.append(round(0.299 * red + 0.587 * green + 0.114 * blue))

    return _to_gray_image(width, height, max_value, gray_values)


def _read_binary_pnm(path: Path, data: bytes) -> GrayImage:
    header_tokens: list[bytes] = []
    i = 0
    while len(header_tokens) < 4 and i < len(data):
        if data[i] in b" \t\r\n":
            i += 1
            continue
        if data[i] == ord("#"):
            while i < len(data) and data[i] not in b"\r\n":
                i += 1
            continue
        start = i
        while i < len(data) and data[i] not in b" \t\r\n":
            i += 1
        header_tokens.append(data[start:i])

    while i < len(data) and data[i] in b" \t\r\n":
        i += 1

    if len(header_tokens) != 4:
        raise ValueError(f"{path} has an incomplete PNM header")

    magic = header_tokens[0]
    width = int(header_tokens[1])
    height = int(header_tokens[2])
    max_value = int(header_tokens[3])
    if max_value <= 0 or max_value > 255:
        raise ValueError(f"{path} must use an 8-bit max value")

    channels = 3 if magic == b"P6" else 1
    expected = width * height * channels
    payload = data[i : i + expected]
    if len(payload) < expected:
        raise ValueError(f"{path} does not contain enough pixel data")

    if magic == b"P5":
        gray_values = list(payload)
    else:
        gray_values = []
        for j in range(0, expected, 3):
            red, green, blue = payload[j], payload[j + 1], payload[j + 2]
            gray_values.append(round(0.299 * red + 0.587 * green + 0.114 * blue))

    return _to_gray_image(width, height, max_value, gray_values)


def _to_gray_image(width: int, height: int, max_value: int, values: list[int]) -> GrayImage:
    scaled = [round((min(max(value, 0), max_value) / max_value) * 255) for value in values]
    rows = [scaled[i : i + width] for i in range(0, width * height, width)]
    return GrayImage(width=width, height=height, pixels=rows)


def write_pgm(path: str | Path, image: GrayImage) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{image.width} {image.height}\n255\n".encode("ascii")
    body = bytes(max(0, min(255, value)) for value in image.flattened())
    path.write_bytes(header + body)
