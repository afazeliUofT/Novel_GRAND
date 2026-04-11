from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from novel_grand.config import run_root
from novel_grand.grand.baselines import run_baseline
from novel_grand.grand.tags_lite import run_tags_grand_lite
from novel_grand.ldpc.bp_trace import BPTraceRunner
from novel_grand.ldpc.features import bit_feature_matrix, oracle_best_snapshot, snapshot_training_rows
from novel_grand.models.training import load_bit_ranker, load_snapshot_selector
from novel_grand.sim.channel import NRSlotQAMLink
from novel_grand.utils.io import ensure_dir, write_jsonl
from novel_grand.utils.seed import set_global_seed


def _safe_bool(x) -> bool:
    return bool(x)


def collect_train_worker(cfg: Dict, worker_id: int, ebn0_db: float) -> Dict:
    seed = int(cfg["system"]["seed"]) + 1000 * worker_id + int(round(ebn0_db * 100))
    set_global_seed(seed)
    torch.set_num_threads(int(cfg["system"]["torch_threads_per_worker"]))

    link = NRSlotQAMLink(cfg)
    tracer = BPTraceRunner(link.encoder, cfg)
    out_root = run_root(cfg)
    shard_dir = ensure_dir(out_root / "train" / "shards")

    n_frames = int(cfg["simulation"]["train_frames_per_worker_per_snr"])
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    nr = cfg["nr"]

    snapshot_rows: List[Dict] = []
    bit_x_rows: List[np.ndarray] = []
    bit_y_rows: List[np.ndarray] = []
    frame_rows: List[Dict] = []

    neg_to_pos = int(cfg["training"]["neg_to_pos_ratio"])
    rng = np.random.default_rng(seed)

    for frame_idx in range(n_frames):
        t0 = time.perf_counter()
        sample = link.sample(ebn0_db)
        trace = tracer.decode_with_trace(sample.llr_ch, sample.codeword_bits, sample.info_bits)
        runtime_ms = 1000.0 * (time.perf_counter() - t0)

        legacy_row = {
            "ebn0_db": float(ebn0_db),
            "worker_id": int(worker_id),
            "frame_idx": int(frame_idx),
            "decoder": "ldpc_only",
            "legacy_detected_failure": int(not trace.legacy_success),
            "selected_snapshot": int(trace.stop_iteration),
            "selected_syndrome_weight": int(trace.snapshots[-1].syndrome_weight),
            "queries": 0,
            "frontier_peak": 0,
            "pattern_weight": 0,
            "success_exact": int(trace.legacy_success and np.array_equal(trace.snapshots[-1].hard, trace.true_codeword)),
            "valid_codeword": int(trace.snapshots[-1].syndrome_mask == 0),
            "undetected_error": int(trace.snapshots[-1].syndrome_mask == 0 and not np.array_equal(trace.snapshots[-1].hard, trace.true_codeword)),
            "primitive_kinds": "",
            "primitive_sizes": "",
            "runtime_ms": float(runtime_ms),
        }
        frame_rows.append(legacy_row)

        if trace.legacy_success:
            continue

        snapshot_rows.extend(snapshot_training_rows(trace, max_iter, tracer.graph_struct.max_vn_degree, tracer.graph_exact.m))

        oracle_idx = oracle_best_snapshot(trace)
        bit_x, bit_y = bit_feature_matrix(
            trace=trace,
            snapshot_idx=oracle_idx,
            max_iter=max_iter,
            max_vn_degree=tracer.graph_struct.max_vn_degree,
            bits_per_symbol=int(nr["bits_per_symbol"]),
            fft_size=int(nr["fft_size"]),
        )
        pos = np.flatnonzero(bit_y > 0.5)
        neg = np.flatnonzero(bit_y <= 0.5)
        if pos.size > 0:
            keep_neg = min(neg.size, neg_to_pos * pos.size)
            keep_neg_idx = rng.choice(neg, size=keep_neg, replace=False) if keep_neg > 0 else np.array([], dtype=int)
            keep = np.concatenate([pos, keep_neg_idx])
            bit_x_rows.append(bit_x[keep])
            bit_y_rows.append(bit_y[keep])

    snapshot_path = shard_dir / f"snapshot_rows_worker{worker_id:02d}_snr{ebn0_db:.2f}.jsonl"
    write_jsonl(snapshot_path, snapshot_rows)

    bit_npz_path = shard_dir / f"bit_rows_worker{worker_id:02d}_snr{ebn0_db:.2f}.npz"
    if bit_x_rows:
        x = np.concatenate(bit_x_rows, axis=0).astype(np.float32)
        y = np.concatenate(bit_y_rows, axis=0).astype(np.float32)
    else:
        x = np.zeros((0, 11), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32)
    np.savez_compressed(bit_npz_path, x=x, y=y)

    frame_path = shard_dir / f"ldpc_frames_worker{worker_id:02d}_snr{ebn0_db:.2f}.jsonl"
    write_jsonl(frame_path, frame_rows)

    return {
        "worker_id": worker_id,
        "ebn0_db": ebn0_db,
        "n_frames": n_frames,
        "n_snapshot_rows": len(snapshot_rows),
        "n_bit_rows": int(x.shape[0]),
        "snapshot_path": str(snapshot_path),
        "bit_npz_path": str(bit_npz_path),
        "frame_path": str(frame_path),
    }


def evaluate_worker(cfg: Dict, worker_id: int, ebn0_db: float) -> Dict:
    seed = int(cfg["system"]["seed"]) + 5000 * worker_id + int(round(ebn0_db * 100))
    set_global_seed(seed)
    torch.set_num_threads(int(cfg["system"]["torch_threads_per_worker"]))

    link = NRSlotQAMLink(cfg)
    tracer = BPTraceRunner(link.encoder, cfg)
    graph_exact = tracer.graph_exact
    graph_struct = tracer.graph_struct
    out_root = run_root(cfg)
    shard_dir = ensure_dir(out_root / "eval" / "shards")
    failure_dir = ensure_dir(out_root / "eval" / "sampled_failures")

    models_dir = out_root / "models"
    snapshot_model = load_snapshot_selector(
        models_dir / "snapshot_selector.pt",
        hidden_dims=cfg["training"]["snapshot_hidden_dims"],
        in_dim=12,
        device=cfg["system"]["device"],
    )
    bit_model = load_bit_ranker(
        models_dir / "bit_ranker.pt",
        hidden_dims=cfg["training"]["bit_hidden_dims"],
        in_dim=11,
        device=cfg["system"]["device"],
    )

    n_frames = int(cfg["simulation"]["eval_frames_per_worker_per_snr"])
    max_failure_samples = int(cfg["simulation"]["sampled_failure_traces_per_worker_per_snr"])
    failure_samples_saved = 0

    baselines = [
        "ldpc_only",
        "final_llr_grand",
        "best_syndrome_llr_grand",
        "best_syndrome_unsat_grand",
        "tags_grand_lite",
    ]
    if cfg["grand"].get("keep_oracle_upper_bound", False):
        baselines.insert(4, "oracle_best_llr")

    frame_rows: List[Dict] = []

    for frame_idx in range(n_frames):
        t0 = time.perf_counter()
        sample = link.sample(ebn0_db)
        trace = tracer.decode_with_trace(sample.llr_ch, sample.codeword_bits, sample.info_bits)
        trace_runtime_ms = 1000.0 * (time.perf_counter() - t0)

        legacy_row = {
            "ebn0_db": float(ebn0_db),
            "worker_id": int(worker_id),
            "frame_idx": int(frame_idx),
            "decoder": "ldpc_only",
            "legacy_detected_failure": int(not trace.legacy_success),
            "selected_snapshot": int(trace.stop_iteration),
            "selected_syndrome_weight": int(trace.snapshots[-1].syndrome_weight),
            "queries": 0,
            "frontier_peak": 0,
            "pattern_weight": 0,
            "success_exact": int(trace.legacy_success and np.array_equal(trace.snapshots[-1].hard, trace.true_codeword)),
            "valid_codeword": int(trace.snapshots[-1].syndrome_mask == 0),
            "undetected_error": int(trace.snapshots[-1].syndrome_mask == 0 and not np.array_equal(trace.snapshots[-1].hard, trace.true_codeword)),
            "primitive_kinds": "",
            "primitive_sizes": "",
            "runtime_ms": float(trace_runtime_ms),
        }
        frame_rows.append(legacy_row)

        if not trace.legacy_success:
            for name in baselines[1:]:
                t1 = time.perf_counter()
                if name == "tags_grand_lite":
                    res = run_tags_grand_lite(trace, graph_exact, graph_struct, snapshot_model, bit_model, cfg)
                else:
                    res = run_baseline(name, trace, graph_exact, cfg)
                res.update(
                    {
                        "ebn0_db": float(ebn0_db),
                        "worker_id": int(worker_id),
                        "frame_idx": int(frame_idx),
                        "legacy_detected_failure": 1,
                        "runtime_ms": 1000.0 * (time.perf_counter() - t1),
                    }
                )
                frame_rows.append(res)

            if failure_samples_saved < max_failure_samples:
                failure_samples_saved += 1
                np.savez_compressed(
                    failure_dir / f"failure_worker{worker_id:02d}_snr{ebn0_db:.2f}_frame{frame_idx:05d}.npz",
                    llr_ch=trace.llr_ch.astype(np.float32),
                    true_codeword=trace.true_codeword.astype(np.uint8),
                    info_bits=trace.info_bits.astype(np.uint8),
                    posterior=np.stack([s.posterior for s in trace.snapshots], axis=0).astype(np.float32),
                    hard=np.stack([s.hard for s in trace.snapshots], axis=0).astype(np.uint8),
                    syndrome_weight=np.array([s.syndrome_weight for s in trace.snapshots], dtype=np.int16),
                    unsat=np.stack([s.unsat_deg for s in trace.snapshots], axis=0).astype(np.int16),
                    flips=np.stack([s.cumulative_flip_count for s in trace.snapshots], axis=0).astype(np.int16),
                )

    frame_path = shard_dir / f"frame_rows_worker{worker_id:02d}_snr{ebn0_db:.2f}.jsonl"
    write_jsonl(frame_path, frame_rows)

    return {
        "worker_id": worker_id,
        "ebn0_db": ebn0_db,
        "n_frames": n_frames,
        "frame_path": str(frame_path),
        "n_rows": len(frame_rows),
    }
