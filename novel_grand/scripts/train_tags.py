from __future__ import annotations

import argparse
import json
import random

import numpy as np

from novel_grand.config import load_config, run_root
from novel_grand.models.training import (
    fit_snapshot_selector,
    fit_template_ranker,
    save_meta_json,
    save_model_bundle,
)
from novel_grand.utils.io import read_jsonl, write_jsonl


PAIR_IN_DIM = 40


def _load_memory_rows(paths):
    rows = []
    for path in paths:
        rows.extend(list(read_jsonl(path)))
    return rows


def _pair_features(q_state, q_risky, q_snap, t_state, t_risky, t_bits, n_snap):
    q = np.asarray(q_state, dtype=np.float32).reshape(-1)
    t = np.asarray(t_state, dtype=np.float32).reshape(-1)
    diff = np.abs(q - t)
    denom = max(float(np.linalg.norm(q) * np.linalg.norm(t)), 1e-6)
    cos = float(np.dot(q, t) / denom)
    qset = set(int(x) for x in q_risky)
    tset = set(int(x) for x in t_risky)
    overlap = float(len(qset & tset)) / max(len(qset | tset), 1)
    tmpl_size = float(len(t_bits)) / 64.0
    snap_gap = abs(int(q_snap) - int(t_state is None or 0))  # placeholder, overwritten below
    return q, t, diff, cos, overlap, tmpl_size


def _build_pair_xy(memory_rows, cfg):
    if not memory_rows:
        return np.zeros((0, PAIR_IN_DIM), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    rng = random.Random(int(cfg["system"]["seed"]) + 17)
    n_snap = int(cfg["legacy_ldpc"]["num_iter"])
    xs = []
    ys = []

    states = [np.asarray(r["state_x"], dtype=np.float32) for r in memory_rows]
    riskies = [list(map(int, r.get("risky_topk", []))) for r in memory_rows]
    snaps = [int(r.get("snapshot_idx", 0)) for r in memory_rows]
    corrs = [list(map(int, r.get("correction_bits", []))) for r in memory_rows]

    def make_feat(i, j):
        q = states[i]
        t = states[j]
        diff = np.abs(q - t)
        denom = max(float(np.linalg.norm(q) * np.linalg.norm(t)), 1e-6)
        cos = float(np.dot(q, t) / denom)
        qset = set(riskies[i])
        tset = set(riskies[j])
        risk_overlap = float(len(qset & tset)) / max(len(qset | tset), 1)
        cset_i = set(corrs[i])
        cset_j = set(corrs[j])
        corr_overlap = float(len(cset_i & cset_j)) / max(len(cset_i | cset_j), 1)
        snap_gap = abs(snaps[i] - snaps[j]) / max(n_snap - 1, 1)
        tmpl_size = float(len(corrs[j])) / 64.0
        return np.concatenate(
            [q, t, diff, np.array([cos, risk_overlap, corr_overlap, tmpl_size, snap_gap, float(abs(memory_rows[i].get("ebn0_db", 0.0) - memory_rows[j].get("ebn0_db", 0.0)) / 5.0)], dtype=np.float32)],
            axis=0,
        )[:PAIR_IN_DIM].astype(np.float32)

    # positives: self and near neighbors with correction overlap
    for i in range(len(memory_rows)):
        xs.append(make_feat(i, i))
        ys.append(1.0)
        sims = []
        for j in range(len(memory_rows)):
            if i == j:
                continue
            cset_i = set(corrs[i])
            cset_j = set(corrs[j])
            corr_overlap = float(len(cset_i & cset_j)) / max(len(cset_i | cset_j), 1)
            if corr_overlap <= 0.0:
                continue
            q = states[i]
            t = states[j]
            denom = max(float(np.linalg.norm(q) * np.linalg.norm(t)), 1e-6)
            cos = float(np.dot(q, t) / denom)
            sims.append((0.7 * corr_overlap + 0.3 * cos, j))
        sims.sort(reverse=True)
        for _, j in sims[:2]:
            xs.append(make_feat(i, j))
            ys.append(1.0)
        neg_pool = [j for j in range(len(memory_rows)) if j != i]
        rng.shuffle(neg_pool)
        added = 0
        for j in neg_pool:
            cset_i = set(corrs[i])
            cset_j = set(corrs[j])
            corr_overlap = float(len(cset_i & cset_j)) / max(len(cset_i | cset_j), 1)
            if corr_overlap > 0.05:
                continue
            xs.append(make_feat(i, j))
            ys.append(0.0)
            added += 1
            if added >= int(cfg["grand"].get("memo_negatives_per_positive", 6)):
                break

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y



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
    memory_paths = sorted(shard_dir.glob("memory_rows_*.jsonl"))
    if not snapshot_paths or not memory_paths:
        raise FileNotFoundError("Training shards not found. Run collect_train first.")

    snap_rows = []
    for path in snapshot_paths:
        snap_rows.extend(list(read_jsonl(path)))
    snap_x = np.asarray([r["x"] for r in snap_rows], dtype=np.float32)
    snap_y = np.asarray([r["y"] for r in snap_rows], dtype=np.float32)

    memory_rows = _load_memory_rows(memory_paths)
    pair_x, pair_y = _build_pair_xy(memory_rows, cfg)
    if pair_x.shape[0] == 0:
        raise RuntimeError("No memory-bank template rows were collected.")

    snap_model, snap_mean, snap_std, snap_meta = fit_snapshot_selector(
        snap_x, snap_y, cfg, device=cfg["system"]["device"]
    )
    tmpl_model, tmpl_mean, tmpl_std, tmpl_meta = fit_template_ranker(
        pair_x, pair_y, cfg, device=cfg["system"]["device"]
    )

    save_model_bundle(model_dir / "snapshot_selector.pt", snap_model, snap_mean, snap_std, snap_meta)
    save_model_bundle(model_dir / "template_ranker.pt", tmpl_model, tmpl_mean, tmpl_std, tmpl_meta)
    write_jsonl(model_dir / "memory_bank.jsonl", memory_rows)
    save_meta_json(
        model_dir / "training_summary.json",
        {
            "snapshot_rows": int(snap_x.shape[0]),
            "memory_rows": int(len(memory_rows)),
            "template_pair_rows": int(pair_x.shape[0]),
            "snapshot_meta": snap_meta,
            "template_meta": tmpl_meta,
        },
    )
    print(json.dumps({"status": "ok", "model_dir": str(model_dir)}, indent=2))


if __name__ == "__main__":
    main()
