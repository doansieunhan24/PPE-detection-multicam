"""Dashboard (CLAUDE.md §7.8): floor map with people coloured by PPE state, alert list, clips.

    python -m src.dashboard.app --results data/results/s01/P --port 8000
    # live (src/analytics_live.py) + OSD video from mediamtx (WebRTC page of path "osd")
    python -m src.dashboard.app --results data/results/live --video-url http://localhost:8889/osd

Reads the alert DB (configs/alerts.yaml → store.sqlite_path, filled by analytics_runner
--db or analytics_live) and the latest PersonPartState per global_id from
<results>/states_latest.json (live) or else <results>/states.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.alerts.store import AlertStore
from src.common.config import DEFAULT_CONFIG_DIR, load_config

STATIC = Path(__file__).parent / "static"


def latest_states(states_path: Path | None, window_ms: int = 2000) -> list[dict]:
    """Last state of every global_id seen within window_ms of the newest timestamp.
    states_path: states.jsonl, a states_latest.json snapshot, or a results dir (snapshot
    preferred, checked on every call since the live runner creates it later)."""
    if states_path is not None and states_path.is_dir():
        snap = states_path / "states_latest.json"
        states_path = snap if snap.exists() else states_path / "states.jsonl"
    if states_path is None or not states_path.exists():
        return []
    last: dict[int, dict] = {}
    if states_path.suffix == ".json":
        try:
            rows = json.loads(states_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []  # being replaced: next poll
        last = {d["global_id"]: d for d in rows}
    else:
        with open(states_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    last[d["global_id"]] = d
    if not last:
        return []
    t = max(d["ts_ms"] for d in last.values())
    return sorted(
        (d for d in last.values() if t - d["ts_ms"] <= window_ms), key=lambda d: d["global_id"]
    )


def create_app(
    store: AlertStore,
    states_path: Path | None = None,
    zones_cfg: dict | None = None,
    evidence_dir: Path | None = None,
    video_url: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Safety Monitoring")
    zones_cfg = zones_cfg or {}

    @app.get("/api/alerts")
    def alerts(limit: int = 100, min_level: int = 1, unacked_only: bool = False) -> list[dict]:
        return store.recent(limit, min_level, unacked_only)

    @app.post("/api/alerts/{alert_id}/ack")
    def ack(alert_id: str) -> dict:
        if not store.ack(alert_id):
            raise HTTPException(404, "unknown alert")
        return {"ok": True}

    @app.get("/api/state")
    def state() -> list[dict]:
        return latest_states(states_path)

    @app.get("/api/config")
    def config() -> dict:
        return {"video_url": video_url}

    @app.get("/api/zones")
    def zones() -> list[dict]:
        return zones_cfg.get("zones", [])

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)  # live clips arrive after start-up
        app.mount("/evidence", StaticFiles(directory=evidence_dir), name="evidence")
    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", help="analytics output dir with states.jsonl")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--session", help="dir whose zones.yaml overrides configs/ (as analytics)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--video-url", help="live OSD page, e.g. mediamtx WebRTC http://host:8889/osd")
    args = ap.parse_args(argv)

    import uvicorn

    cfg_dir = Path(args.config)
    alerts_cfg = load_config("alerts", cfg_dir)
    store = AlertStore(alerts_cfg["store"]["sqlite_path"])
    app = create_app(
        store,
        Path(args.results) if args.results else None,
        load_config("zones", cfg_dir, override_dir=args.session),
        Path(alerts_cfg["smart_record"]["out_dir"]),
        args.video_url,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
