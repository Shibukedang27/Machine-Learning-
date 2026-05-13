"""Small local web app for stock pattern analysis."""

from __future__ import annotations

import argparse
import html
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data_tools import generate_demo_stock
from .model import analyze, load_model, train

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "stocks" / "demo_stock.csv"
MODEL_PATH = ROOT / "models" / "stock_pattern_model.json"
UPLOAD_DIR = ROOT / "data" / "stock_uploads"


class StockAppHandler(BaseHTTPRequestHandler):
    server_version = "StockPatternAnalysis/1.0"

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
            generate_demo_stock(DATA_PATH)
            self._redirect("/?message=Demo+stock+data+generated")
            return
        if path == "/train":
            train(DATA_PATH, MODEL_PATH)
            self._redirect("/?message=Model+trained")
            return
        if path == "/analyze":
            message = self._handle_analyze()
            self._redirect(f"/?message={message}")
            return
        self.send_error(404)

    def _handle_analyze(self) -> str:
        csv_path = DATA_PATH
        fields = parse_multipart(self)
        uploaded = fields.get("stock_csv")
        if uploaded and uploaded["filename"]:
            if not str(uploaded["filename"]).lower().endswith(".csv"):
                return "Please+upload+a+CSV+file"
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
            csv_path.write_bytes(uploaded["content"])
        result = analyze(csv_path, load_model(MODEL_PATH))
        return f"{result.pattern.replace(' ', '+')}:+{result.probability_up:.0%}+up+probability"

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
  <title>Stock Pattern Analysis</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #eef5f3; color: #202124; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 34px 20px 52px; }}
    header {{ display: grid; grid-template-columns: 1.05fr .95fr; gap: 28px; align-items: center; margin-bottom: 22px; }}
    h1 {{ font-size: clamp(2.2rem, 4vw, 4.7rem); line-height: 1; margin: 0 0 14px; letter-spacing: 0; }}
    p {{ color: #53605a; line-height: 1.6; font-size: 1rem; }}
    .chart {{ height: 292px; border-radius: 8px; background: #102a43; position: relative; overflow: hidden; box-shadow: 0 20px 54px rgba(21, 35, 50, .18); }}
    .chart svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    section {{ background: #ffffff; border: 1px solid #d7e3df; border-radius: 8px; padding: 18px; min-height: 178px; }}
    h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
    .metric {{ color: #0f766e; font-size: 2rem; font-weight: 800; margin: 8px 0; }}
    button, input::file-selector-button {{ border: 0; border-radius: 6px; background: #0f766e; color: #fff; padding: 10px 14px; font-weight: 800; cursor: pointer; }}
    button.blue {{ background: #2855c7; }}
    input[type=file] {{ width: 100%; margin: 10px 0 14px; }}
    .message {{ margin-bottom: 18px; background: #edfdf7; border-color: #9ee6c9; color: #075f4f; }}
    .fine {{ font-size: .92rem; color: #6a746e; }}
    @media (max-width: 820px) {{ header, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Stock Pattern Analysis</h1>
        <p>A local machine-learning project that studies price, volume, trend, volatility, and range behavior to explain the latest chart setup in plain language.</p>
        <p class="fine">Educational project only. It is not financial advice.</p>
      </div>
      <div class="chart" aria-hidden="true">
        <svg viewBox="0 0 600 320">
          <defs><linearGradient id="fill" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#2dd4bf" stop-opacity=".44"/><stop offset="1" stop-color="#2dd4bf" stop-opacity="0"/></linearGradient></defs>
          <path d="M0 260 L48 238 L96 246 L144 210 L192 218 L240 175 L288 188 L336 132 L384 148 L432 102 L480 116 L528 72 L600 88 L600 320 L0 320 Z" fill="url(#fill)"/>
          <path d="M0 260 L48 238 L96 246 L144 210 L192 218 L240 175 L288 188 L336 132 L384 148 L432 102 L480 116 L528 72 L600 88" fill="none" stroke="#2dd4bf" stroke-width="6"/>
          <g stroke="#f7c948" stroke-width="4"><line x1="86" y1="276" x2="86" y2="228"/><line x1="222" y1="218" x2="222" y2="160"/><line x1="438" y1="142" x2="438" y2="84"/><line x1="530" y1="108" x2="530" y2="54"/></g>
        </svg>
      </div>
    </header>
    {f'<section class="message">{html.escape(message)}</section>' if message else ''}
    <div class="grid">
      <section>
        <h2>Data</h2>
        <div class="metric">{summary['rows']}</div>
        <p>Rows in the current demo CSV.</p>
        <form method="post" action="/generate-demo"><button type="submit">Generate Demo Data</button></form>
      </section>
      <section>
        <h2>Model</h2>
        <div class="metric">{summary['model']}</div>
        <p>Trains a compact logistic classifier on chart features.</p>
        <form method="post" action="/train"><button class="blue" type="submit">Train Model</button></form>
      </section>
      <section>
        <h2>Analyze</h2>
        <p>Use the demo data or upload your own CSV with date, open, high, low, close, and volume columns.</p>
        <form method="post" action="/analyze" enctype="multipart/form-data">
          <input type="file" name="stock_csv" accept=".csv">
          <button type="submit">Analyze Pattern</button>
        </form>
      </section>
    </div>
  </main>
</body>
</html>"""


def current_summary() -> dict[str, str]:
    rows = 0
    if DATA_PATH.exists():
        rows = max(0, len(DATA_PATH.read_text(encoding="utf-8").splitlines()) - 1)
    model = "Not trained"
    if MODEL_PATH.exists():
        data = load_model(MODEL_PATH)
        model = f"{data['training_accuracy']:.0%}"
    return {"rows": str(rows), "model": model}


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
    class Handler(StockAppHandler):
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
    parser = argparse.ArgumentParser(description="Run the stock pattern analysis web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    run(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
