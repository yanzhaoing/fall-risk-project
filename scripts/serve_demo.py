#!/usr/bin/env python3
"""Serve the competition demo dashboard without third-party web dependencies."""

from __future__ import annotations

import cgi
import json
import shutil
import tempfile
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.demo_engine import SCENARIOS, demo_engine
from src.inference.video_analyzer import analyze_video_file, build_sample_analysis


HOST = "127.0.0.1"
PORT = 7860


class DemoHandler(BaseHTTPRequestHandler):
    """Small HTTP handler for the demo UI and JSON API."""

    def log_message(self, fmt: str, *args) -> None:
        print(f"[demo] {self.address_string()} - {fmt % args}")

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _readiness_payload(self) -> dict:
        root = ROOT
        results = root / "results"
        docs = root / "docs"
        artifacts = {
            "demo_evaluation": (results / "demo_evaluation.json").exists(),
            "public_ntu_evaluation": (results / "public_ntu_evaluation.json").exists(),
            "submission_readiness": (results / "submission_readiness.json").exists(),
            "minimal_user_materials": (docs / "minimal_user_materials.md").exists(),
            "evidence_matrix": (docs / "evidence_matrix.md").exists(),
        }
        return {
            "artifacts": artifacts,
            "next_needed": [
                "真实居家场景视频或截图",
                "萤石开放平台截图或接口日志",
                "团队与报名信息",
            ],
        }

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _read_uploaded_video(self) -> tuple[str, str]:
        """Parse a multipart upload and persist the video to a temp file."""
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 75 * 1024 * 1024:
            raise RuntimeError("uploaded video is too large")

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
        }
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
            keep_blank_values=True,
        )
        field = form["video"]
        if isinstance(field, list):
            field = field[0]
        if not getattr(field, "filename", None):
            raise RuntimeError("missing video file")

        suffix = Path(field.filename).suffix or ".mp4"
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=ROOT / "results")
        with temp:
            shutil.copyfileobj(field.file, temp)
        return temp.name, field.filename

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (ROOT / "web" / "templates" / "dashboard.html").read_bytes()
            self._send_bytes(html, "text/html; charset=utf-8")
            return

        if path == "/api/demo/scenarios":
            self._send_json({
                "scenarios": [
                    {"id": key, **value}
                    for key, value in SCENARIOS.items()
                ]
            })
            return

        if path == "/api/demo/sequence":
            self._send_json({"items": demo_engine.run_sequence()})
            return

        if path == "/api/video/sample":
            self._send_json(build_sample_analysis())
            return

        if path == "/api/project/readiness":
            self._send_json(self._readiness_payload())
            return

        if path.startswith("/static/"):
            name = path.removeprefix("/static/")
            file_path = ROOT / "web" / "static" / name
            if file_path.exists() and file_path.is_file():
                content_type = "text/css; charset=utf-8" if name.endswith(".css") else "application/javascript; charset=utf-8"
                self._send_bytes(file_path.read_bytes(), content_type)
                return

        if path.startswith("/artifacts/"):
            rel = path.removeprefix("/artifacts/")
            file_path = ROOT / rel
            if file_path.exists() and file_path.is_file():
                if file_path.suffix == ".json":
                    content_type = "application/json; charset=utf-8"
                else:
                    content_type = "text/plain; charset=utf-8"
                self._send_bytes(file_path.read_bytes(), content_type)
                return

        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/demo/evaluate":
            payload = self._read_json()
            scenario = payload.get("scenario", "normal")
            self._send_json(demo_engine.evaluate(scenario).__dict__)
            return

        if path == "/api/video/analyze":
            upload_path = None
            original_name = None
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" in content_type:
                    upload_path, original_name = self._read_uploaded_video()
                    payload = analyze_video_file(upload_path)
                    payload["summary"]["source_label"] = original_name
                    self._send_json(payload)
                    return

                payload = self._read_json()
                if payload.get("mode") == "sample" or not payload:
                    self._send_json(build_sample_analysis())
                    return

                self._send_json({"error": "unsupported request body"}, status=400)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            finally:
                if upload_path:
                    try:
                        Path(upload_path).unlink(missing_ok=True)
                    except OSError:
                        pass
            return

        self._send_json({"error": "not found"}, status=404)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Demo dashboard: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
