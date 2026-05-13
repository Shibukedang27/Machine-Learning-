"""Local web app for WiFi motion radar analysis."""

from __future__ import annotations

import argparse
import html
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .model import load_model, predict, train, training_accuracy
from .signal_data import generate_demo_dataset, read_wifi_csv

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "wifi" / "demo_wifi_signals.csv"
MODEL_PATH = ROOT / "models" / "wifi_motion_model.json"
UPLOAD_DIR = ROOT / "data" / "wifi_uploads"


class WifiAppHandler(BaseHTTPRequestHandler):
    server_version = "WifiMotionRadar/1.0"

    def do_HEAD(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        page = render_home().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/":
            self._send_html(render_home())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/generate-demo":
            generate_demo_dataset(DATA_PATH)
            self._redirect("/?message=Demo+WiFi+signal+data+generated")
            return
        if path == "/train":
            train(DATA_PATH, MODEL_PATH)
            self._redirect("/?message=WiFi+motion+model+trained")
            return
        if path == "/predict":
            message = self._handle_predict()
            self._redirect(f"/?message={message}")
            return
        self.send_error(404)

    def _handle_predict(self) -> str:
        csv_path = DATA_PATH
        fields = parse_multipart(self)
        uploaded = fields.get("wifi_csv")
        if uploaded and uploaded["filename"]:
            if not str(uploaded["filename"]).lower().endswith(".csv"):
                return "Please+upload+a+CSV+file"
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
            csv_path.write_bytes(uploaded["content"])
        result = predict(csv_path, load_model(MODEL_PATH))
        return f"{result.room_state.replace(' ', '+')}:+{result.label}+({result.confidence:.0%}+confidence)"

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()


def render_home() -> str:
    message = parse_qs(urlparse(getattr(_RequestContext, "path", "")).query).get("message", [""])[0]
    summary = current_summary()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WiFi Motion Radar</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #eef4f8; color: #1e252b; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 34px 20px 52px; }}
    header {{ display: grid; grid-template-columns: 1.05fr .95fr; gap: 28px; align-items: center; margin-bottom: 22px; }}
    h1 {{ font-size: clamp(2.2rem, 4vw, 4.7rem); line-height: 1; margin: 0 0 14px; letter-spacing: 0; }}
    p {{ color: #50616f; line-height: 1.6; font-size: 1rem; }}
    .radar {{ height: 292px; border-radius: 8px; background: #101820; position: relative; overflow: hidden; box-shadow: 0 20px 54px rgba(12, 20, 28, .18); }}
    .radar svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    section {{ background: #ffffff; border: 1px solid #d6e1e8; border-radius: 8px; padding: 18px; min-height: 178px; }}
    h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
    .metric {{ color: #0f766e; font-size: 2rem; font-weight: 800; margin: 8px 0; }}
    button, input::file-selector-button {{ border: 0; border-radius: 6px; background: #0f766e; color: #fff; padding: 10px 14px; font-weight: 800; cursor: pointer; }}
    button.blue {{ background: #2855c7; }}
    input[type=file] {{ width: 100%; margin: 10px 0 14px; }}
    .message {{ margin-bottom: 18px; background: #edfdf7; border-color: #9ee6c9; color: #075f4f; }}
    .fine {{ font-size: .92rem; color: #667580; }}
    @media (max-width: 820px) {{ header, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>WiFi Motion Radar</h1>
        <p>A local machine-learning project that treats WiFi signal changes like a small radar trace, then estimates room movement and the closest consent-based motion profile.</p>
        <p class="fine">Demo data only. Real-world use must be visible, consent-based, and privacy respecting.</p>
      </div>
      <div class="radar" aria-hidden="true">
        <svg viewBox="0 0 600 320">
          <circle cx="300" cy="160" r="116" fill="none" stroke="#2dd4bf" stroke-opacity=".25" stroke-width="3"/>
          <circle cx="300" cy="160" r="78" fill="none" stroke="#2dd4bf" stroke-opacity=".28" stroke-width="3"/>
          <circle cx="300" cy="160" r="38" fill="none" stroke="#2dd4bf" stroke-opacity=".32" stroke-width="3"/>
          <path d="M300 160 L493 82" stroke="#2dd4bf" stroke-width="8" stroke-linecap="round"/>
          <path d="M300 160 C250 108 215 94 171 112" fill="none" stroke="#f7c948" stroke-width="5"/>
          <path d="M300 160 C356 214 399 232 452 205" fill="none" stroke="#4ade80" stroke-width="5"/>
          <circle cx="171" cy="112" r="9" fill="#f7c948"/>
          <circle cx="452" cy="205" r="9" fill="#4ade80"/>
        </svg>
      </div>
    </header>
    {f'<section class="message">{html.escape(message)}</section>' if message else ''}
    <div class="grid">
      <section>
        <h2>Signal Data</h2>
        <div class="metric">{summary['samples']}</div>
        <p>Synthetic CSI-like samples in the current demo file.</p>
        <form method="post" action="/generate-demo"><button type="submit">Generate Demo Data</button></form>
      </section>
      <section>
        <h2>Model</h2>
        <div class="metric">{summary['model']}</div>
        <p>Builds motion fingerprints from signal-window features.</p>
        <form method="post" action="/train"><button class="blue" type="submit">Train Model</button></form>
      </section>
      <section>
        <h2>Identify Motion</h2>
        <p>Use the demo data or upload a WiFi signal CSV with timestamp, label, and subcarrier columns.</p>
        <form method="post" action="/predict" enctype="multipart/form-data">
          <input type="file" name="wifi_csv" accept=".csv">
          <button type="submit">Analyze Room</button>
        </form>
      </section>
    </div>
  </main>
</body>
</html>"""


def current_summary() -> dict[str, str]:
    samples = "0"
    if DATA_PATH.exists():
        samples = str(len(read_wifi_csv(DATA_PATH)))
    model = "Not trained"
    if MODEL_PATH.exists() and DATA_PATH.exists():
        model_data = load_model(MODEL_PATH)
        model = f"{training_accuracy(DATA_PATH, model_data):.0%}"
    return {"samples": samples, "model": model}


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
        name = _header_value(disposition, "name")
        filename = _header_value(disposition, "filename")
        if name:
            fields[name] = {"filename": filename, "content": content.rstrip(b"\r\n")}
    return fields


def _header_value(header: str, key: str) -> str:
    marker = f'{key}="'
    if marker not in header:
        return ""
    return header.split(marker, 1)[1].split('"', 1)[0]


class _RequestContext:
    path = ""


def run(host: str, port: int) -> None:
    class Handler(WifiAppHandler):
        def do_GET(self) -> None:
            _RequestContext.path = self.path
            super().do_GET()

        def do_HEAD(self) -> None:
            _RequestContext.path = self.path
            super().do_HEAD()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Running on http://{host}:{port}")
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the WiFi motion radar web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    run(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
