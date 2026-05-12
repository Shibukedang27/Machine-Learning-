"""Tiny dependency-free web app for local face recognition demos."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .demo_data import generate_demo_dataset
from .recognizer import load_model, recognize, train

ROOT = Path.cwd()
DATASET_DIR = ROOT / "data" / "people"
MODEL_PATH = ROOT / "models" / "face_model.json"
UPLOAD_DIR = ROOT / "data" / "uploads"


class FaceAppHandler(BaseHTTPRequestHandler):
    server_version = "FaceRecognitionSystem/1.0"

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path != "/":
            self.send_error(404)
            return
        encoded = render_home().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(render_home())
            return
        if path.startswith("/static/"):
            self._send_static(path.removeprefix("/static/"))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/generate-demo":
            generate_demo_dataset(DATASET_DIR)
            self._redirect("/?message=Demo+dataset+generated")
            return
        if path == "/train":
            train(DATASET_DIR, MODEL_PATH)
            self._redirect("/?message=Model+trained")
            return
        if path == "/recognize":
            message = self._handle_recognize()
            self._redirect(f"/?message={message}")
            return
        self.send_error(404)

    def _handle_recognize(self) -> str:
        fields = parse_multipart(self)
        uploaded = fields.get("face_image")
        if not uploaded or not uploaded["filename"]:
            return "Please+choose+a+PGM+or+PPM+image"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded["filename"]).suffix.lower()
        if suffix not in {".pgm", ".ppm"}:
            return "Only+PGM+and+PPM+images+are+supported"
        image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        image_path.write_bytes(uploaded["content"])
        prediction = recognize(load_model(MODEL_PATH), image_path)
        return f"Prediction:+{prediction.label}+({prediction.confidence:.0%}+confidence)"

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_static(self, relative_path: str) -> None:
        path = ROOT / "docs" / "screenshots" / relative_path
        if not path.exists():
            self.send_error(404)
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()


def parse_multipart(handler: BaseHTTPRequestHandler) -> dict[str, dict[str, bytes | str]]:
    content_type = handler.headers.get("Content-Type", "")
    if "boundary=" not in content_type:
        return {}
    boundary = content_type.split("boundary=", 1)[1].encode("utf-8")
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length)
    fields: dict[str, dict[str, bytes | str]] = {}

    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", errors="replace")
        disposition = next((line for line in headers.splitlines() if line.lower().startswith("content-disposition")), "")
        name = _extract_header_value(disposition, "name")
        filename = _extract_header_value(disposition, "filename")
        if name:
            fields[name] = {"filename": filename, "content": content.rstrip(b"\r\n")}
    return fields


def _extract_header_value(header: str, key: str) -> str:
    marker = f'{key}="'
    if marker not in header:
        return ""
    return header.split(marker, 1)[1].split('"', 1)[0]


def render_home() -> str:
    query = parse_qs(urlparse(getattr(_RequestContext, "path", "")).query)
    message = query.get("message", [""])[0]
    people = dataset_summary()
    model = model_summary()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Recognition System</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f4f7fb; color: #17202a; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    header {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 28px; align-items: center; margin-bottom: 28px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4.5rem); line-height: 1; margin: 0 0 16px; letter-spacing: 0; }}
    p {{ color: #4d5d6c; font-size: 1rem; line-height: 1.6; }}
    .hero-art {{ min-height: 280px; border-radius: 8px; background: linear-gradient(135deg, #0f766e, #1d4ed8); display: grid; place-items: center; box-shadow: 0 20px 50px rgba(17, 24, 39, .16); }}
    .face {{ width: 180px; height: 220px; border-radius: 48% 48% 44% 44%; background: #f8d6b3; position: relative; box-shadow: inset 0 -16px 0 rgba(92, 57, 33, .08); }}
    .face:before, .face:after {{ content: ""; position: absolute; top: 74px; width: 20px; height: 20px; border-radius: 50%; background: #17202a; }}
    .face:before {{ left: 48px; }} .face:after {{ right: 48px; }}
    .mouth {{ position: absolute; left: 58px; bottom: 54px; width: 64px; height: 26px; border-bottom: 6px solid #17202a; border-radius: 0 0 64px 64px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    section, .panel {{ background: #ffffff; border: 1px solid #dbe3ee; border-radius: 8px; padding: 18px; }}
    h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
    button, input::file-selector-button {{ border: 0; border-radius: 6px; background: #0f766e; color: white; padding: 10px 14px; font-weight: 700; cursor: pointer; }}
    button.secondary {{ background: #1d4ed8; }}
    input[type=file] {{ width: 100%; margin: 10px 0 14px; }}
    .metric {{ font-size: 2rem; font-weight: 800; color: #0f766e; margin: 8px 0; }}
    .message {{ background: #ecfdf5; border-color: #a7f3d0; color: #065f46; margin-bottom: 18px; }}
    @media (max-width: 820px) {{ header, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Face Recognition System</h1>
        <p>Train and test a local face recognizer with labeled images. The demo uses synthetic face-like samples so the full workflow can be tested without collecting personal biometric data.</p>
      </div>
      <div class="hero-art" aria-hidden="true"><div class="face"><div class="mouth"></div></div></div>
    </header>
    {f'<section class="message">{html.escape(message)}</section>' if message else ''}
    <div class="grid">
      <section>
        <h2>Dataset</h2>
        <div class="metric">{sum(people.values())}</div>
        <p>{html.escape(json.dumps(people, sort_keys=True))}</p>
        <form method="post" action="/generate-demo"><button type="submit">Generate Demo Data</button></form>
      </section>
      <section>
        <h2>Training</h2>
        <div class="metric">{model}</div>
        <p>Builds one normalized profile per registered person.</p>
        <form method="post" action="/train"><button class="secondary" type="submit">Train Model</button></form>
      </section>
      <section>
        <h2>Recognition</h2>
        <p>Upload a cropped `.pgm` or `.ppm` face image and compare it with the trained profiles.</p>
        <form method="post" action="/recognize" enctype="multipart/form-data">
          <input type="file" name="face_image" accept=".pgm,.ppm" required>
          <button type="submit">Recognize Face</button>
        </form>
      </section>
    </div>
  </main>
</body>
</html>"""


class _RequestContext:
    path = ""


def dataset_summary() -> dict[str, int]:
    if not DATASET_DIR.exists():
        return {}
    return {
        path.name: len([item for item in path.iterdir() if item.suffix.lower() in {".pgm", ".ppm"}])
        for path in DATASET_DIR.iterdir()
        if path.is_dir()
    }


def model_summary() -> str:
    if not MODEL_PATH.exists():
        return "Not trained"
    model = load_model(MODEL_PATH)
    return f"{len(model['profiles'])} people"


def run(host: str, port: int) -> None:
    class Handler(FaceAppHandler):
        def do_GET(self) -> None:
            _RequestContext.path = self.path
            super().do_GET()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Running on http://{host}:{port}")
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local face recognition web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
