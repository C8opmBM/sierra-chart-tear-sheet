"""Minimal local web app for iterative tear-sheet generation.

Serves the most recently generated report and exposes an upload endpoint
that lets you drop in a new Trade Activity Log and regenerate the report in
place — see the "Upload New Log" button embedded in the report itself
(rendered when render_report(..., live_upload=True)).

Run directly:
    python -m tearsheet.webapp

Or via the bundled Dockerfile/docker-compose.yml (see repo root).

Configuration (environment variables):
    TEARSHEET_INPUT_DIR   directory uploaded logs are saved to (default /input)
    TEARSHEET_OUTPUT_DIR  directory the report is written to  (default /output)
    PORT                  port to listen on                   (default 8080)
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request, send_file, Response

from tearsheet.app.main import run

UPLOAD_DIR = Path(os.environ.get("TEARSHEET_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("TEARSHEET_OUTPUT_DIR", "/output"))
REPORT_PATH = OUTPUT_DIR / "report.html"
TRADES_CSV_PATH = OUTPUT_DIR / "trades.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

_PLACEHOLDER_HTML = """<!doctype html>
<html>
<head>
  <title>Sierra Chart Tear Sheet</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;">
  <div style="text-align:center;max-width:360px;">
    <h2 style="margin-bottom:4px;">No report yet</h2>
    <p style="color:#8b949e;font-size:0.9rem;margin-top:0;">Upload a Trade Activity Log file to generate one.</p>
    <form id="f" style="text-align:left;margin-top:20px;">
      <div style="margin-bottom:10px;">
        <label style="display:block;font-size:0.8rem;margin-bottom:4px;">Trade Activity Log file</label>
        <input type="file" name="file" required style="width:100%;">
      </div>
      <div style="margin-bottom:10px;">
        <label style="display:block;font-size:0.8rem;margin-bottom:4px;">Starting balance (optional)</label>
        <input type="number" step="any" name="starting_balance" style="width:100%;padding:6px;background:#161b22;color:inherit;border:1px solid #30363d;border-radius:4px;">
      </div>
      <div style="margin-bottom:16px;">
        <label style="display:block;font-size:0.8rem;margin-bottom:4px;">Risk capital (optional)</label>
        <input type="number" step="any" name="risk_capital" style="width:100%;padding:6px;background:#161b22;color:inherit;border:1px solid #30363d;border-radius:4px;">
      </div>
      <button type="submit" style="width:100%;padding:8px;background:#58a6ff;color:#0d1117;border:none;border-radius:6px;font-weight:600;cursor:pointer;">Generate</button>
    </form>
    <div id="status" style="margin-top:10px;font-size:0.85rem;"></div>
  </div>
  <script>
    document.getElementById('f').addEventListener('submit', function(e) {
      e.preventDefault();
      var status = document.getElementById('status');
      var btn = e.target.querySelector('button[type="submit"]');
      status.style.color = '';
      status.textContent = 'Processing…';
      btn.disabled = true;
      fetch('/upload', { method: 'POST', body: new FormData(e.target) })
        .then(function(r) {
          if (!r.ok) { return r.text().then(function(t) { throw new Error(t || r.statusText); }); }
          return r.text();
        })
        .then(function() { window.location.reload(); })
        .catch(function(err) {
          status.style.color = '#f85149';
          status.textContent = 'Error: ' + err.message;
          btn.disabled = false;
        });
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index() -> Response:
    if REPORT_PATH.exists():
        return send_file(REPORT_PATH)
    return Response(_PLACEHOLDER_HTML, mimetype="text/html")


def _optional_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _param(form_value: str | None, env_var: str) -> float | None:
    """Form value wins when given; otherwise fall back to an env-var default.

    Lets a deployment bake in fixed account parameters (e.g. a specific
    prop-firm account's starting balance/risk capital/drawdown rules) via
    environment variables, while still allowing a per-upload override
    through the form when needed.
    """
    explicit = _optional_float(form_value)
    if explicit is not None:
        return explicit
    return _optional_float(os.environ.get(env_var))


@app.route("/upload", methods=["POST"])
def upload() -> Response:
    f = request.files.get("file")
    if f is None or f.filename == "":
        return Response("No file uploaded.", status=400)

    starting_balance = _param(request.form.get("starting_balance"), "TEARSHEET_STARTING_BALANCE")
    risk_capital = _param(request.form.get("risk_capital"), "TEARSHEET_RISK_CAPITAL")
    drawdown_limit = _param(request.form.get("drawdown_limit"), "TEARSHEET_DRAWDOWN_LIMIT")
    daily_loss_limit = _param(request.form.get("daily_loss_limit"), "TEARSHEET_DAILY_LOSS_LIMIT")
    profit_target = _param(request.form.get("profit_target"), "TEARSHEET_PROFIT_TARGET")

    # Sanitize the filename ourselves rather than trusting the client;
    # werkzeug's secure_filename isn't imported to keep this dependency-light,
    # so just strip path separators.
    safe_name = os.path.basename(f.filename).replace("..", "_")
    dest = UPLOAD_DIR / safe_name
    f.save(dest)

    try:
        run(
            dest,
            REPORT_PATH,
            starting_balance=starting_balance,
            risk_capital=risk_capital,
            drawdown_limit=drawdown_limit,
            daily_loss_limit=daily_loss_limit,
            profit_target=profit_target,
            live_upload=True,
            # Always emit a per-trade CSV next to the report — cheap, and
            # gives you a ready-made file to diff against an independently
            # kept trading journal without needing any CLI overrides.
            export_trades_csv=TRADES_CSV_PATH,
        )
    except Exception as exc:  # noqa: BLE001 - surface the real error to the uploader
        return Response(f"Failed to generate report: {exc}", status=500)

    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
