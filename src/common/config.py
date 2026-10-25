"""Load YAML configs from configs/ (CLAUDE.md D8: no hardcoded thresholds)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


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


SESSION_CONFIGS = ("cameras", "zones")  # may differ per recording session; thresholds may not
