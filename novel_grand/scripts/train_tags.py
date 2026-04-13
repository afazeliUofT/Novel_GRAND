from __future__ import annotations

import argparse
import json

import numpy as np

from novel_grand.config import load_config, run_root
from novel_grand.models.maskdiff import fit_maskdiff, save_maskdiff_bundle
from novel_grand.models.training import (
    fit_snapshot_selector,
    save_meta_json,
    save_model_bundle,
)
from novel_grand.utils.io import read_jsonl



def _load_maskdiff_rows(paths):
    token_xs = []
    global_xs = []
    ys = []
    ms = []
    for path in paths:
        data = np.load(path)
        if data["token_x"].shape[0] == 0:
            continue
        token_xs.append(data["token_x"].astype(np.float32))
        global_xs.append(data["global_x"].astype(np.float32))
        ys.append(data["y"].astype(np.float32))
        ms.append(data["loss_mask"].astype(np.float32))
    if not token_xs:
        return None, None, None, None
    return (
        np.concatenate(token_xs, axis=0),
        np.concatenate(global_xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ms, axis=0),
    )



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
    maskdiff_paths = sorted(shard_dir.glob("maskdiff_rows_*.npz"))
    if not snapshot_paths or not maskdiff_paths:
        raise FileNotFoundError("Training shards not found. Run collect_train first.")

    snap_rows = []
    for path in snapshot_paths:
        snap_rows.extend(list(read_jsonl(path)))
    snap_x = np.asarray([r["x"] for r in snap_rows], dtype=np.float32)
    snap_y = np.asarray([r["y"] for r in snap_rows], dtype=np.float32)

    token_x, global_x, yy, mm = _load_maskdiff_rows(maskdiff_paths)
    if token_x is None:
        raise RuntimeError("No masked-diffusion training rows were collected.")

    snap_model, snap_mean, snap_std, snap_meta = fit_snapshot_selector(
        snap_x, snap_y, cfg, device=cfg["system"]["device"]
    )
    md_model, tok_mean, tok_std, glb_mean, glb_std, md_meta = fit_maskdiff(
        token_x, global_x, yy, mm, cfg, device=cfg["system"]["device"]
    )

    save_model_bundle(model_dir / "snapshot_selector.pt", snap_model, snap_mean, snap_std, snap_meta)
    save_maskdiff_bundle(model_dir / "maskdiff.pt", md_model, tok_mean, tok_std, glb_mean, glb_std, md_meta)
    save_meta_json(
        model_dir / "training_summary.json",
        {
            "snapshot_rows": int(snap_x.shape[0]),
            "maskdiff_rows": int(token_x.shape[0]),
            "snapshot_meta": snap_meta,
            "maskdiff_meta": md_meta,
        },
    )
    print(json.dumps({"status": "ok", "model_dir": str(model_dir)}, indent=2))


if __name__ == "__main__":
    main()
