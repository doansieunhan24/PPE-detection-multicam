"""Run analytics offline from PerceptionFrame JSONL (CLAUDE.md D4: replayable).

    python -m src.analytics_runner --jsonl data/jsonl/session01/ --out data/results/session01/P
    python -m src.analytics_runner --jsonl ... --out ... --policy B3
    python -m src.analytics_runner --jsonl ... --out ... --ablation use_visibility=false

cameras.yaml / zones.yaml inside the --jsonl dir (written by tools/session_check.py) override
the ones in --config for that session; all thresholds always come from --config.

Writes states.jsonl, alerts.jsonl, gid_map.jsonl (ts_ms, cam_id, track_id, global_id),
run.json (config snapshot, git commit, counts).

obs_cache (eval only): association, part observations (p, v) and behavior alerts do not depend
on the policy or on the fusion.yaml sections in POLICY_ONLY_SECTIONS, so they are computed once
per session/config and replayed for every policy / threshold (same output, much faster).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

from src.alerts.notifier_telegram import TelegramNotifier
from src.alerts.store import AlertStore
from src.association.ground_plane import sync_groups
from src.behavior.engine import BehaviorEngine
from src.common.config import (
    DEFAULT_CONFIG_DIR,
    SESSION_CONFIGS,
    apply_overrides,
    load_config,
)
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging
from src.fusion.engine import PPEEngine
from src.fusion.policies import POLICY_IDS
from src.geometry.homography import CameraCalibration, load_calibration

log = get_logger(__name__)
# fusion.yaml sections read only by the policy / alert stage (PPEEngine.decide)
POLICY_ONLY_SECTIONS = ("evidence", "alert", "baselines")


def git_commit() -> str | None:
    """Short HEAD hash, suffixed -dirty when tracked files have uncommitted changes."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL, check=False
            ).returncode
            != 0
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


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


class Analytics:
    """PPE fusion + behavior for one synchronized group of frames (shared by the offline
    runner and src/analytics_live.py, so live and replay give the same alerts)."""

    def __init__(
        self,
        config_dir: Path,
        session_dir: Path | None,
        policy: str = "P",
        ablation: dict | None = None,
        fusion_overrides: dict | None = None,
    ) -> None:
        self.fusion_cfg = apply_overrides(load_config("fusion", config_dir), fusion_overrides)
        self.zones_cfg = load_config("zones", config_dir, override_dir=session_dir)
        self.cams_cfg = load_config("cameras", config_dir, override_dir=session_dir)
        self.session_configs = [
            n for n in SESSION_CONFIGS if session_dir and (session_dir / f"{n}.yaml").exists()
        ]
        if self.session_configs:
            log.info("session config from %s: %s", session_dir, self.session_configs)
        self.alerts_cfg = load_config("alerts", config_dir)
        behavior_cfg = self.behavior_cfg = load_config("behavior", config_dir)
        self.calibs = load_calibs(self.cams_cfg, config_dir.parent)
        levels = self.alerts_cfg.get("levels", {})
        self.engine = PPEEngine(
            self.fusion_cfg, self.zones_cfg, self.calibs, policy, ablation or {}, levels
        )
        self.behavior = BehaviorEngine(behavior_cfg, self.zones_cfg, levels)

    def step(self, group: dict, ts: int):
        states, alerts, gid_of = self.engine.step(group, ts)
        floor_of = {st.global_id: st.floor_xy for st in states}
        alerts += self.behavior.step(group, ts, gid_of, floor_of)
        return states, alerts, gid_of


def _file_sig(path: str | Path | None) -> list | None:
    if not path or not Path(path).exists():
        return None
    st = Path(path).stat()
    return [str(path), st.st_size, st.st_mtime_ns]


def obs_cache_key(jsonl_paths: list[Path], an: Analytics, ablation: dict) -> str:
    """Hash of everything PPEEngine.observe and the behavior engine depend on."""
    f = {k: v for k, v in an.fusion_cfg.items() if k not in POLICY_ONLY_SECTIONS}
    payload = [
        [_file_sig(p) for p in jsonl_paths],
        f,
        _file_sig(f.get("part_assign", {}).get("p_calibration")),
        _file_sig(f.get("visibility", {}).get("learned_path")),
        bool(ablation.get("v_learned")),
        an.zones_cfg,
        an.cams_cfg,
        an.behavior_cfg,
        an.alerts_cfg.get("levels", {}),
        sorted((c, cal.calibrated) for c, cal in an.calibs.items()),
    ]
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def run(
    jsonl_dir: Path,
    out_dir: Path,
    config_dir: Path,
    policy: str,
    ablation: dict[str, bool],
    db: bool = False,
    fusion_overrides: dict | None = None,
    cams: list[str] | None = None,
    obs_cache: Path | None = None,
) -> dict:
    """fusion_overrides: dotted keys into fusion.yaml (eval/run_sensitivity.py).
    cams: use only these cameras' JSONL (camera-count experiment, eval/run_num_cams.py).
    obs_cache: dir for the policy-independent cache (see module doc)."""
    an = Analytics(config_dir, jsonl_dir, policy, ablation, fusion_overrides)
    fusion_cfg, cams_cfg, calibs = an.fusion_cfg, an.cams_cfg, an.calibs
    alerts_cfg, overrides = an.alerts_cfg, an.session_configs

    paths = [p for p in sorted(jsonl_dir.glob("*.jsonl")) if cams is None or p.stem in cams]
    if cams is not None and set(cams) - {p.stem for p in paths}:
        raise SystemExit(f"{jsonl_dir}: no JSONL for {sorted(set(cams) - {p.stem for p in paths})}")
    if not paths:
        raise SystemExit(f"no *.jsonl in {jsonl_dir}")
    cache_file = None
    cached = None
    if obs_cache is not None:
        cache_file = Path(obs_cache) / f"{jsonl_dir.name}_{obs_cache_key(paths, an, ablation)}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)

    def steps():
        """(ts, observed, gid_of, behavior alerts) per synced group."""
        if cached is not None:
            yield from cached
            return
        streams = {p.stem: sorted(read_frames(p), key=lambda f: f.ts_ms) for p in paths}
        rec = [] if cache_file is not None else None
        for group in sync_groups(streams, cams_cfg.get("sync_tol_ms", 50)):
            ts = min(f.ts_ms for f in group.values())
            observed, gid_of = an.engine.observe(group, ts)
            floor_of = {o.gid: o.floor for o in observed}
            beh = an.behavior.step(group, ts, gid_of, floor_of)
            item = (ts, observed, gid_of, beh)
            if rec is not None:
                rec.append(item)
            yield item
        if rec is not None:  # atomic: parallel eval jobs may build the same cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_file)

    store = AlertStore(alerts_cfg["store"]["sqlite_path"]) if db else None
    notifier = TelegramNotifier.from_config(alerts_cfg) if db else None

    out_dir.mkdir(parents=True, exist_ok=True)
    n_groups = n_alerts = 0
    t_first = t_last = None
    with (
        open(out_dir / "states.jsonl", "w", encoding="utf-8") as fs,
        open(out_dir / "alerts.jsonl", "w", encoding="utf-8") as fa,
        open(out_dir / "gid_map.jsonl", "w", encoding="utf-8") as fg,
    ):
        for ts, observed, gid_of, beh in steps():
            t_first = ts if t_first is None else t_first
            t_last = ts
            states, alerts = an.engine.decide(ts, observed)
            alerts = alerts + beh
            for s in states:
                fs.write(s.model_dump_json() + "\n")
            for a in alerts:
                fa.write(a.model_dump_json() + "\n")
                if store is not None:
                    store.add(a)
                if notifier is not None:
                    notifier.send(a)
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
        "fusion_overrides": fusion_overrides or {},
        "commit": git_commit(),
        "cams": sorted(p.stem for p in paths),
        "session_configs": overrides,
        "calibrated_cams": sorted(c for c, cal in calibs.items() if cal.calibrated),
        "groups": n_groups,
        "alerts": n_alerts,
        "duration_s": ((t_last - t_first) / 1000) if n_groups else 0.0,
        "fusion_cfg": fusion_cfg,
        "obs_cache": None if cache_file is None else ("hit" if cached else "built"),
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
    ap.add_argument("--db", action="store_true", help="also store alerts in SQLite (+Telegram)")
    ap.add_argument("--cams", nargs="+", help="camera subset (default: every <cam>.jsonl)")
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
        db=args.db,
        cams=args.cams,
    )


if __name__ == "__main__":
    main()
