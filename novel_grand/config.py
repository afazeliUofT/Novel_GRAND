from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping in config file: {path}")
    cfg["config_path"] = str(path)
    cfg["repo_root"] = str(path.parent.parent.resolve())
    return cfg


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def run_root(cfg: Dict[str, Any]) -> Path:
    repo_root = Path(cfg["repo_root"])
    output_root = Path(cfg.get("output_root", "outputs"))
    exp = cfg["experiment_name"]
    out = (repo_root / output_root / exp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out
