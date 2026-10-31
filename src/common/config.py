"""Load YAML configs from configs/ (CLAUDE.md D8: no hardcoded thresholds)."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_ENV_FILE = DEFAULT_CONFIG_DIR.parent / ".env"
_ENV_REF = re.compile(r"\$\{(\w+)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    name: str, config_dir: str | Path = DEFAULT_CONFIG_DIR, override_dir: str | Path | None = None
) -> dict[str, Any]:
    """Load configs/<name>.yaml, e.g. load_config("fusion").

    override_dir: if <override_dir>/<name>.yaml exists it is used instead (per-session
    cameras.yaml / zones.yaml next to the session's JSONL).
    """
    if override_dir is not None and (Path(override_dir) / f"{name}.yaml").exists():
        return load_yaml(Path(override_dir) / f"{name}.yaml")
    return load_yaml(Path(config_dir) / f"{name}.yaml")


def read_env_file(path: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines of a .env file (no dependency on python-dotenv, absent in the DS image)."""
    env: dict[str, str] = {}
    if not Path(path).exists():
        return env
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def expand_uri(uri: str, env: dict[str, str]) -> str:
    """Replace ${VAR} in a camera URI, URL-encoding the value (passwords may contain '@')."""

    def sub(m: re.Match) -> str:
        if m.group(1) not in env:
            raise KeyError(f"{m.group(1)} not set (.env or environment) for camera uri")
        return quote(env[m.group(1)], safe="")

    return _ENV_REF.sub(sub, uri)


def load_cameras(path: str | Path, env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, Any]:
    """cameras.yaml with ${VAR} in each uri filled from the environment, then .env.

    Credentials stay in .env (gitignored), never in configs/.
    """
    cfg = load_yaml(path)
    env = {**read_env_file(env_file), **os.environ}
    for cam in cfg.get("cameras", []):
        cam["uri"] = expand_uri(cam["uri"], env)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Copy of cfg with dotted keys set, e.g. {"evidence.theta_obs": 0.5}. Keys must exist."""
    out = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        node = out
        *path, leaf = key.split(".")
        for k in path:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[leaf] = value
    return out


SESSION_CONFIGS = ("cameras", "zones")  # may differ per recording session; thresholds may not
