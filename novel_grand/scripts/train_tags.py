from __future__ import annotations

import argparse
import json

import numpy as np

from novel_grand.config import load_config, run_root
from novel_grand.models.training import (
    fit_action_prior,
    fit_snapshot_selector,
    fit_state_value,
    save_meta_json,
    save_model_bundle,
)
from novel_grand.utils.io import read_jsonl



def _load_xy_rows(paths):
    xs = []
    ys = []
    for path in paths:
        data = np.load(path)
        if data["x"].shape[0] == 0:
            continue
        xs.append(data["x"].astype(np.float32))
        ys.append(data["y"].astype(np.float32))
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    shard_dir = out_root / "train" / "shards"
    model_dir = out_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths = sorted(shard_dir.glob("snapshot_rows_*.jsonl"))
    action_paths = sorted(shard_dir.glob("action_rows_*.npz"))
    value_paths = sorted(shard_dir.glob("value_rows_*.npz"))
    if not snapshot_paths or not action_paths or not value_paths:
        raise FileNotFoundError("Training shards not found. Run collect_train first.")

    snap_rows = []
    for path in snapshot_paths:
        snap_rows.extend(list(read_jsonl(path)))
    snap_x = np.asarray([r["x"] for r in snap_rows], dtype=np.float32)
    snap_y = np.asarray([r["y"] for r in snap_rows], dtype=np.float32)

    action_x, action_y = _load_xy_rows(action_paths)
    value_x, value_y = _load_xy_rows(value_paths)
    if action_x is None or value_x is None:
        raise RuntimeError("No GFlow/TTA training rows were collected.")

    snap_model, snap_mean, snap_std, snap_meta = fit_snapshot_selector(
        snap_x, snap_y, cfg, device=cfg["system"]["device"]
    )
    act_model, act_mean, act_std, act_meta = fit_action_prior(
        action_x, action_y, cfg, device=cfg["system"]["device"]
    )
    val_model, val_mean, val_std, val_meta = fit_state_value(
        value_x, value_y, cfg, device=cfg["system"]["device"]
    )

    save_model_bundle(model_dir / "snapshot_selector.pt", snap_model, snap_mean, snap_std, snap_meta)
    save_model_bundle(model_dir / "action_prior.pt", act_model, act_mean, act_std, act_meta)
    save_model_bundle(model_dir / "state_value.pt", val_model, val_mean, val_std, val_meta)
    save_meta_json(
        model_dir / "training_summary.json",
        {
            "snapshot_rows": int(snap_x.shape[0]),
            "action_rows": int(action_x.shape[0]),
            "value_rows": int(value_x.shape[0]),
            "snapshot_meta": snap_meta,
            "action_meta": act_meta,
            "value_meta": val_meta,
        },
    )
    print(json.dumps({"status": "ok", "model_dir": str(model_dir)}, indent=2))


if __name__ == "__main__":
    main()
