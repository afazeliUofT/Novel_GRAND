from __future__ import annotations

import argparse
import json

import numpy as np

from novel_grand.config import load_config, run_root
from novel_grand.models.training import (
    fit_bit_ranker,
    fit_snapshot_selector,
    save_meta_json,
    save_model_bundle,
)
from novel_grand.utils.io import read_jsonl


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
    bit_paths = sorted(shard_dir.glob("bit_rows_*.jsonl"))
    if not snapshot_paths or not bit_paths:
        raise FileNotFoundError("Training shards not found. Run collect_train first.")

    snap_rows = []
    for path in snapshot_paths:
        snap_rows.extend(list(read_jsonl(path)))
    bit_rows = []
    for path in bit_paths:
        bit_rows.extend(list(read_jsonl(path)))

    snap_x = np.asarray([r["x"] for r in snap_rows], dtype=np.float32)
    snap_y = np.asarray([r["y"] for r in snap_rows], dtype=np.float32)
    bit_x = np.asarray([r["x"] for r in bit_rows], dtype=np.float32)
    bit_y = np.asarray([r["y"] for r in bit_rows], dtype=np.float32)

    snap_model, snap_mean, snap_std, snap_meta = fit_snapshot_selector(
        snap_x, snap_y, cfg, device=cfg["system"]["device"]
    )
    bit_model, bit_mean, bit_std, bit_meta = fit_bit_ranker(
        bit_x, bit_y, cfg, device=cfg["system"]["device"]
    )

    save_model_bundle(model_dir / "snapshot_selector.pt", snap_model, snap_mean, snap_std, snap_meta)
    save_model_bundle(model_dir / "bit_ranker.pt", bit_model, bit_mean, bit_std, bit_meta)
    save_meta_json(
        model_dir / "training_summary.json",
        {
            "snapshot_rows": int(snap_x.shape[0]),
            "bit_rows": int(bit_x.shape[0]),
            "snapshot_meta": snap_meta,
            "bit_meta": bit_meta,
        },
    )
    print(json.dumps({"status": "ok", "model_dir": str(model_dir)}, indent=2))


if __name__ == "__main__":
    main()
