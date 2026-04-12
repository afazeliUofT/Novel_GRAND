from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _discover_repo_root(start: Path) -> Path:
    """Find the repo root robustly.

    Preferred order:
    1) any ancestor containing pyproject.toml
    2) any ancestor containing .git
    3) fallback to the historical parent.parent behavior
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    # Historical fallback. This is intentionally last because it caused
    # nested outputs when configs lived under outputs/.../reports/.
    if start.parent != start:
        return start.parent.parent.resolve()
    return start



def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping in config file: {path}")

    cfg["config_path"] = str(path)

    repo_root_value = cfg.get("repo_root", None)
    if repo_root_value:
        repo_root = Path(repo_root_value).expanduser().resolve()
    else:
        repo_root = _discover_repo_root(path.parent)
    cfg["repo_root"] = str(repo_root)
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
