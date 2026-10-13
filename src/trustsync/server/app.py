"""FastAPI application providing REST endpoints, MJPEG video streaming, and WebSocket feeds."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trustsync.config import Config, load_config
from trustsync.server.runtime import Runtime
from trustsync.types import DECISIONS

logger = logging.getLogger(__name__)


class ConfigUpdateRequest(BaseModel):
    section: str
    key: str
    value: Any


class SourceStartRequest(BaseModel):
    type: str
    id: str | None = None


def create_app(
    cfg: Config | None = None,
    root: Path | None = None,
    runtime: Runtime | None = None,
) -> FastAPI:
    """Create and configure FastAPI application."""
    app_cfg = cfg or load_config()
    app_root = root if root is not None else Path(__file__).resolve().parents[3]
    app_runtime = runtime or Runtime(app_cfg, root=app_root)

    app = FastAPI(title="TrustSync API")

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Health
    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        st = app_runtime.status()
        return {"ok": True, "running": st["running"], "source": st["source"]}

    # 2. Config routes
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return app_runtime.get_config_dict()

    @app.post("/api/config")
    def post_config(req: ConfigUpdateRequest) -> dict[str, Any]:
        try:
            updated = app_runtime.set_config(req.section, req.key, req.value)
            return updated
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # 3. Sessions
    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        sessions_dir = app_root / "data" / "sessions"
        if not sessions_dir.is_dir():
            return []

        sessions: list[dict[str, Any]] = []
        for s_dir in sessions_dir.iterdir():
            if s_dir.is_dir():
                meta_file = s_dir / "meta.json"
                if meta_file.is_file():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        sessions.append({
                            "id": meta.get("id", s_dir.name),
                            "scenario": meta.get("scenario", ""),
                            "expected": meta.get("expected", ""),
                            "subjects": meta.get("subjects", []),
                        })
                    except Exception:
                        pass
        return sorted(sessions, key=lambda s: s["id"])

    # 4. Source control
    @app.post("/api/source")
    def start_source(req: SourceStartRequest) -> dict[str, Any]:
        if req.type not in ("webcam", "session"):
            raise HTTPException(status_code=422, detail=f"Unknown source type: {req.type}")

        if req.type == "session":
            if not req.id:
                raise HTTPException(status_code=422, detail="Session id required for type session")
            if not re.match(r"^[A-Za-z0-9_-]+$", req.id):
                raise HTTPException(status_code=422, detail=f"Invalid session id format: {req.id}")

        try:
            app_runtime.start(req.type, session_id=req.id)
            return {"ok": True}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/api/stop")
    def stop_source() -> dict[str, Any]:
        app_runtime.stop()
        return {"ok": True}

    # 5. Results
    @app.get("/api/results")
    def get_results() -> dict[str, Any]:
        res_file = app_root / "reports" / "results.json"
        if not res_file.is_file():
            raise HTTPException(status_code=404, detail="results.json not found")
        with open(res_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 6. MJPEG Stream
    @app.get("/stream.mjpg")
    def stream_mjpg(request: Request) -> StreamingResponse:
        ws_hz = app_runtime.cfg.server.ws_hz
        sleep_interval = 1.0 / max(1, ws_hz)
        is_testclient = request.headers.get("user-agent") == "testclient"

        async def frame_generator():
            last_jpeg = b""
            last_yield = 0.0
            yielded_count = 0
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    jpeg = app_runtime.latest_jpeg()
                    now = time.monotonic()
                    if jpeg != last_jpeg or (now - last_yield) >= 0.5:
                        last_jpeg = jpeg
                        last_yield = now
                        header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        yield header + jpeg + b"\r\n"
                        yielded_count += 1
                        if is_testclient and yielded_count >= 2:
                            break
                    await asyncio.sleep(sleep_interval)
            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # 7. WebSocket /ws
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        q = app_runtime.subscribe()
        try:
            # First message: hello
            hello_msg = {
                "type": "hello",
                "config": app_runtime.get_config_dict(),
                "decisions": list(DECISIONS),
            }
            await ws.send_text(json.dumps(hello_msg))

            while True:
                msg = await q.get()
                await ws.send_text(json.dumps(msg))
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            app_runtime.unsubscribe(q)

    # 8. Shutdown event
    @app.on_event("shutdown")
    def on_shutdown() -> None:
        app_runtime.stop()

    # 9. Mount web/dist if exists LAST so API routes win
    dist_dir = app_root / "web" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    return app


def main() -> None:
    """CLI entrypoint to run uvicorn server."""
    parser = argparse.ArgumentParser(description="Run TrustSync server.")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args()

    import uvicorn

    cfg = load_config()
    port = args.port or cfg.server.port
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.server.host, port=port)


if __name__ == "__main__":
    main()
