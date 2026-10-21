"""Dashboard (CLAUDE.md §7.8): floor map with people coloured by PPE state, alert list, clips.

    python -m src.dashboard.app --results data/results/s01/P --port 8000

Reads the alert DB (configs/alerts.yaml → store.sqlite_path, filled by analytics_runner
--db) and the latest PersonPartState per global_id from <results>/states.jsonl.
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
    """Last state of every global_id seen within window_ms of the newest timestamp."""
    if states_path is None or not states_path.exists():
        return []
    last: dict[int, dict] = {}
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

    @app.get("/api/zones")
    def zones() -> list[dict]:
        return zones_cfg.get("zones", [])

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    if evidence_dir is not None and evidence_dir.exists():
        app.mount("/evidence", StaticFiles(directory=evidence_dir), name="evidence")
    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", help="analytics output dir with states.jsonl")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    import uvicorn

    cfg_dir = Path(args.config)
    alerts_cfg = load_config("alerts", cfg_dir)
    store = AlertStore(alerts_cfg["store"]["sqlite_path"])
    states = Path(args.results) / "states.jsonl" if args.results else None
    app = create_app(
        store, states, load_config("zones", cfg_dir), Path(alerts_cfg["smart_record"]["out_dir"])
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
