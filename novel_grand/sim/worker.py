from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from novel_grand.config import run_root
from novel_grand.grand.baselines import run_baseline, run_baseline_detailed, run_teacher_best_snapshot_llr
from novel_grand.grand.maskdiff import run_maskdiff_grand
from novel_grand.grand.tags_lite import run_selector_llr_grand
from novel_grand.ldpc.bp_trace import BPTraceRunner
from novel_grand.ldpc.features import teacher_snapshot_training_rows
from novel_grand.ldpc.maskdiff_features import build_maskdiff_training_rows
from novel_grand.models.maskdiff import load_maskdiff
from novel_grand.models.training import load_snapshot_selector
from novel_grand.sim.channel import NRSlotQAMLink
from novel_grand.utils.io import ensure_dir, write_jsonl
from novel_grand.utils.seed import set_global_seed


MASKDIFF_TOKEN_DIM = 16
MASKDIFF_GLOBAL_DIM = 17



def collect_train_worker(cfg: Dict, worker_id: int, ebn0_db: float) -> Dict:
    seed = int(cfg["system"]["seed"]) + 1000 * worker_id + int(round(ebn0_db * 100))
    set_global_seed(seed)
    torch.set_num_threads(int(cfg["system"]["torch_threads_per_worker"]))

    link = NRSlotQAMLink(cfg, seed=seed)
    tracer = BPTraceRunner(link.encoder, cfg)
    out_root = run_root(cfg)
    shard_dir = ensure_dir(out_root / "train" / "shards")

    n_frames = int(cfg["simulation"]["train_frames_per_worker_per_snr"])
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    rng = np.random.default_rng(seed)

    snapshot_rows: List[Dict] = []
    mdiff_token_rows: List[np.ndarray] = []
    mdiff_global_rows: List[np.ndarray] = []
    mdiff_y_rows: List[np.ndarray] = []
    mdiff_mask_rows: List[np.ndarray] = []
    frame_rows: List[Dict] = []

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
            "query_budget": 0,
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

        guard = run_baseline_detailed("final_llr_grand", trace, tracer.graph_exact, cfg)
        if bool(guard.get("success_exact", False)):
            # The AI stage only sees post-guard failures.
            continue

        teacher_cap = int(cfg["grand"].get("ai_teacher_cap", cfg["grand"].get("rescue_bonus_cap", cfg["grand"]["query_cap"])))
        teacher = run_teacher_best_snapshot_llr(trace, tracer.graph_exact, cfg, query_cap=teacher_cap)
        teacher_exact = bool(teacher.get("success_exact", False))
        teacher_snapshot_idx = int(teacher.get("selected_snapshot_index", len(trace.snapshots) - 1))

        # Teacher-aligned snapshot selector rows.
        snapshot_rows.extend(
            teacher_snapshot_training_rows(
                trace,
                teacher_snapshot_idx,
                max_iter,
                tracer.graph_struct.max_vn_degree,
                tracer.graph_exact.m,
            )
        )

        target_mask = None
        if teacher_exact:
            target_mask = (
                trace.snapshots[teacher_snapshot_idx].hard.astype(np.uint8)
                ^ np.asarray(teacher["corrected_bits"], dtype=np.uint8)
            ).astype(np.float32)

        tok_x, glb_x, yy, mm = build_maskdiff_training_rows(
            trace=trace,
            snapshot_idx=teacher_snapshot_idx,
            graph_exact=tracer.graph_exact,
            graph_struct=tracer.graph_struct,
            cfg=cfg,
            target_mask=target_mask,
            rng=rng,
        )
        if tok_x.shape[0] > 0:
            mdiff_token_rows.append(tok_x)
            mdiff_global_rows.append(glb_x)
            mdiff_y_rows.append(yy)
            mdiff_mask_rows.append(mm)

    snapshot_path = shard_dir / f"snapshot_rows_worker{worker_id:02d}_snr{ebn0_db:.2f}.jsonl"
    write_jsonl(snapshot_path, snapshot_rows)

    mdiff_npz_path = shard_dir / f"maskdiff_rows_worker{worker_id:02d}_snr{ebn0_db:.2f}.npz"
    topk_bits = int(cfg["grand"].get("mdiff_shortlist_bits", 64))
    if mdiff_token_rows:
        tx = np.concatenate(mdiff_token_rows, axis=0).astype(np.float32)
        gx = np.concatenate(mdiff_global_rows, axis=0).astype(np.float32)
        yy = np.concatenate(mdiff_y_rows, axis=0).astype(np.float32)
        mm = np.concatenate(mdiff_mask_rows, axis=0).astype(np.float32)
    else:
        tx = np.zeros((0, topk_bits, MASKDIFF_TOKEN_DIM), dtype=np.float32)
        gx = np.zeros((0, MASKDIFF_GLOBAL_DIM), dtype=np.float32)
        yy = np.zeros((0, topk_bits), dtype=np.float32)
        mm = np.zeros((0, topk_bits), dtype=np.float32)
    np.savez_compressed(mdiff_npz_path, token_x=tx, global_x=gx, y=yy, loss_mask=mm)

    frame_path = shard_dir / f"ldpc_frames_worker{worker_id:02d}_snr{ebn0_db:.2f}.jsonl"
    write_jsonl(frame_path, frame_rows)

    return {
        "worker_id": worker_id,
        "ebn0_db": ebn0_db,
        "n_frames": n_frames,
        "n_snapshot_rows": len(snapshot_rows),
        "n_maskdiff_rows": int(tx.shape[0]),
        "snapshot_path": str(snapshot_path),
        "maskdiff_npz_path": str(mdiff_npz_path),
        "frame_path": str(frame_path),
    }



def evaluate_worker(cfg: Dict, worker_id: int, ebn0_db: float) -> Dict:
    seed = int(cfg["system"]["seed"]) + 5000 * worker_id + int(round(ebn0_db * 100))
    set_global_seed(seed)
    torch.set_num_threads(int(cfg["system"]["torch_threads_per_worker"]))

    link = NRSlotQAMLink(cfg, seed=seed)
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
    maskdiff_model = load_maskdiff(
        models_dir / "maskdiff.pt",
        device=cfg["system"]["device"],
    )

    n_frames = int(cfg["simulation"]["eval_frames_per_worker_per_snr"])
    max_failure_samples = int(cfg["simulation"]["sampled_failure_traces_per_worker_per_snr"])
    failure_samples_saved = 0

    baselines = [
        "ldpc_only",
        "final_llr_grand",
        "final_llr_grand_capmatched",
        "guard_plus_best_syndrome",
        "selector_llr_grand",
        "maskdiff_grand",
    ]
    if cfg["grand"].get("keep_oracle_upper_bound", False):
        baselines.append("oracle_best_llr")

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
            "query_budget": 0,
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
                if name == "selector_llr_grand":
                    res = run_selector_llr_grand(trace, graph_exact, graph_struct, snapshot_model, cfg)
                elif name == "maskdiff_grand":
                    res = run_maskdiff_grand(trace, graph_exact, graph_struct, snapshot_model, maskdiff_model, cfg)
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
