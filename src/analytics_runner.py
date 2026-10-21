"""Run analytics offline from PerceptionFrame JSONL (CLAUDE.md D4: replayable).

    python -m src.analytics_runner --jsonl data/jsonl/session01/ --out data/results/session01/P
    python -m src.analytics_runner --jsonl ... --out ... --policy B3
    python -m src.analytics_runner --jsonl ... --out ... --ablation use_visibility=false

Writes states.jsonl, alerts.jsonl, gid_map.jsonl (ts_ms, cam_id, track_id, global_id),
run.json (config snapshot, git commit, counts).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.association.ground_plane import sync_groups
from src.common.config import DEFAULT_CONFIG_DIR, load_config
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.fusion.engine import PPEEngine
from src.fusion.policies import POLICY_IDS
from src.geometry.homography import CameraCalibration, load_calibration

log = get_logger(__name__)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_ablation(items: list[str]) -> dict[str, bool | str]:
    out: dict[str, bool | str] = {}
    for it in items:
        k, _, v = it.partition("=")
        low = v.lower()
        if low in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            out[k] = low in ("1", "true", "yes", "on")
        else:
            out[k] = v
    return out


def load_calibs(cams_cfg: dict, root: Path) -> dict[str, CameraCalibration]:
    out = {}
    for c in cams_cfg.get("cameras", []):
        path = c.get("calibration")
        if path and (root / path).exists():
            out[c["cam_id"]] = load_calibration(root / path)
    return out


def run(
    jsonl_dir: Path,
    out_dir: Path,
    config_dir: Path,
    policy: str,
    ablation: dict[str, bool],
) -> dict:
    fusion_cfg = load_config("fusion", config_dir)
    zones_cfg = load_config("zones", config_dir)
    cams_cfg = load_config("cameras", config_dir)
    alerts_cfg = load_config("alerts", config_dir)
    calibs = load_calibs(cams_cfg, config_dir.parent)

    streams = {
        p.stem: sorted(read_frames(p), key=lambda f: f.ts_ms)
        for p in sorted(jsonl_dir.glob("*.jsonl"))
    }
    if not streams:
        raise SystemExit(f"no *.jsonl in {jsonl_dir}")
    engine = PPEEngine(
        fusion_cfg, zones_cfg, calibs, policy, ablation, alerts_cfg.get("levels", {})
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_groups = n_alerts = 0
    t_first = t_last = None
    with (
        open(out_dir / "states.jsonl", "w", encoding="utf-8") as fs,
        open(out_dir / "alerts.jsonl", "w", encoding="utf-8") as fa,
        open(out_dir / "gid_map.jsonl", "w", encoding="utf-8") as fg,
    ):
        for group in sync_groups(streams, cams_cfg.get("sync_tol_ms", 50)):
            ts = min(f.ts_ms for f in group.values())
            t_first = ts if t_first is None else t_first
            t_last = ts
            states, alerts, gid_of = engine.step(group, ts)
            for s in states:
                fs.write(s.model_dump_json() + "\n")
            for a in alerts:
                fa.write(a.model_dump_json() + "\n")
            for (cam, tid), gid in gid_of.items():
                fg.write(
                    json.dumps({"ts_ms": ts, "cam_id": cam, "track_id": tid, "global_id": gid})
                    + "\n"
                )
            n_groups += 1
            n_alerts += len(alerts)

    summary = {
        "jsonl": str(jsonl_dir),
        "policy": policy,
        "ablation": ablation,
        "commit": git_commit(),
        "cams": sorted(streams),
        "calibrated_cams": sorted(c for c, cal in calibs.items() if cal.calibrated),
        "groups": n_groups,
        "alerts": n_alerts,
        "duration_s": ((t_last - t_first) / 1000) if n_groups else 0.0,
        "fusion_cfg": fusion_cfg,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("%s: %d groups, %d alerts → %s", policy, n_groups, n_alerts, out_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", required=True, help="dir with <cam_id>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--policy", default="P", choices=POLICY_IDS)
    ap.add_argument(
        "--ablation",
        nargs="*",
        default=[],
        help="use_visibility=false temporal=false allow_unknown=false method=dempster_shafer",
    )
    args = ap.parse_args(argv)
    setup_logging()
    run(
        Path(args.jsonl),
        Path(args.out),
        Path(args.config),
        args.policy,
        _parse_ablation(args.ablation),
    )


if __name__ == "__main__":
    main()
