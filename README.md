# Machine Learning Projects

This repository contains hands-on machine learning projects that run locally and keep the setup simple. The code is written to be readable first, so each project can be understood, tested, and improved without needing a heavy framework.

## Projects

- Face Recognition System
- Stock Pattern Analysis

---

# Face Recognition System

A pure-Python face recognition learning project that runs locally with no external services. It includes a small web app, command-line tools, generated demo face data, unit tests, and a simple ML-style embedding classifier.

This project is intended for consent-based classroom or portfolio use. Do not use it for surveillance, secret identification, or recognizing people who have not agreed to participate.

## Features

- Register people from local `.pgm` or `.ppm` face images
- Train a nearest-profile face classifier from labeled folders
- Recognize a new face image from the command line or browser
- Generate demo face-like images for safe testing
- Run without third-party Python packages
- Includes automated tests

## Project Structure

```text
src/face_recognition_system/
  app.py          Local web interface
  cli.py          Command-line entry point
  demo_data.py    Synthetic demo face generator
  image_io.py     PGM/PPM image reader and writer
  recognizer.py   Embedding and recognition logic
tests/
  test_image_io.py
  test_recognizer.py
docs/screenshots/
  app-preview.svg
```

## Quick Start

Generate demo data:

```bash
python3 -m src.face_recognition_system.cli generate-demo --output data/people
```

Train the model:

```bash
python3 -m src.face_recognition_system.cli train --dataset data/people --model models/face_model.json
```

Recognize a demo image:

```bash
python3 -m src.face_recognition_system.cli recognize --model models/face_model.json --image data/people/Ada/ada_01.pgm
```

Run the web app:

```bash
python3 -m src.face_recognition_system.app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Image Format

The system accepts portable graymap/pixmap images:

- `.pgm` grayscale images
- `.ppm` RGB images

These formats keep the project dependency-free. You can convert images to PGM with common image tools, or use the generated demo data.

## How It Works

1. Each image is converted to grayscale.
2. The face crop is resized into a compact fixed-size vector.
3. Brightness is normalized to reduce lighting differences.
4. The system stores an average embedding for each person.
5. Recognition compares a new embedding with known profiles using cosine similarity.

This is a lightweight educational recognizer, not a production-grade biometric model.

## Responsible Use

- Get clear consent from every person in the dataset.
- Store data locally and delete it when it is no longer needed.
- Do not use this to make high-stakes decisions.
- Test on diverse lighting and camera conditions before relying on results.

---

# Stock Pattern Analysis

A local stock-pattern analysis project that uses machine learning to study chart behavior from OHLCV data: open, high, low, close, and volume. It trains a compact logistic classifier on engineered features such as returns, moving-average distance, volatility, volume pressure, and price position inside the recent range.

The output is written in plain language, for example: bullish breakout setup, trend pullback, bearish continuation risk, sideways consolidation, or mixed pattern.

This project is for learning and portfolio work only. It is not financial advice, and it should not be used as the only reason to buy or sell anything.

## Stock Project Features

- Generate a demo stock CSV
- Train a small ML model from price and volume behavior
- Analyze the latest chart setup
- Upload your own CSV in the browser app
- Run with the Python standard library only
- Includes automated tests

## Stock Quick Start

Generate demo stock data:

```bash
python3 -m src.stock_pattern_analysis.cli generate-demo --output data/stocks/demo_stock.csv
```

Train the model:

```bash
python3 -m src.stock_pattern_analysis.cli train --csv data/stocks/demo_stock.csv --model models/stock_pattern_model.json
```

Analyze the latest pattern:

```bash
python3 -m src.stock_pattern_analysis.cli analyze --csv data/stocks/demo_stock.csv --model models/stock_pattern_model.json
```

Run the web app:

```bash
python3 -m src.stock_pattern_analysis.app --host 127.0.0.1 --port 8010
```

Then open:

```text
http://127.0.0.1:8010
```

## CSV Format

Use a CSV with these columns:

```text
date,open,high,low,close,volume
```

The app expects at least 25 rows so it can calculate trend and volatility features properly.

## Testing

Run all tests:

```bash
python3 -m unittest discover -s tests
```
