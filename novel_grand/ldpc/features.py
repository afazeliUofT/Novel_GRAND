from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from novel_grand.ldpc.bp_trace import Snapshot, TraceResult


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a).astype(np.uint8) ^ np.asarray(b).astype(np.uint8)))


def oracle_best_snapshot(trace: TraceResult) -> int:
    """Oracle snapshot by raw Hamming distance to the true codeword.

    This remains useful as a diagnostic upper bound, but it is not the same
    objective as rescue-within-budget. The updated training pipeline therefore
    prefers teacher-aligned labels.
    """
    dists = [hamming_distance(s.hard, trace.true_codeword) for s in trace.snapshots]
    return int(np.argmin(dists))


def snapshot_feature_vector(snapshot: Snapshot, max_iter: int, max_vn_degree: int, n: int, m: int) -> np.ndarray:
    post = snapshot.posterior
    abs_post = np.abs(post)
    low_abs = abs_post < 1.0
    return np.array(
        [
            snapshot.iter_idx / max(max_iter, 1),
            snapshot.syndrome_weight / max(m, 1),
            float(abs_post.mean()),
            float(abs_post.min(initial=0.0)),
            float(np.quantile(abs_post, 0.1)),
            float(np.quantile(abs_post, 0.9)),
            float(snapshot.unsat_deg.mean()) / max(max_vn_degree, 1),
            float(snapshot.unsat_deg.max(initial=0)) / max(max_vn_degree, 1),
            float(snapshot.change_fraction),
            float((snapshot.cumulative_flip_count > 0).mean()),
            float(low_abs.mean()),
            float(np.count_nonzero(snapshot.hard)) / max(n, 1),
        ],
        dtype=np.float32,
    )


def bit_feature_matrix(
    trace: TraceResult,
    snapshot_idx: int,
    max_iter: int,
    max_vn_degree: int,
    bits_per_symbol: int,
    fft_size: int,
    target_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    snapshot = trace.snapshots[snapshot_idx]
    post = snapshot.posterior.astype(np.float32)
    llr_ch = trace.llr_ch.astype(np.float32)
    abs_post = np.abs(post)
    abs_llr = np.abs(llr_ch)
    unsat = snapshot.unsat_deg.astype(np.float32)
    flips = snapshot.cumulative_flip_count.astype(np.float32)
    n = post.size
    idx = np.arange(n, dtype=np.float32)

    same_symbol_mean = np.zeros(n, dtype=np.float32)
    same_subcarrier_mean = np.zeros(n, dtype=np.float32)

    if bits_per_symbol > 0:
        num_symbols = n // bits_per_symbol
        symbol_abs = abs_post.reshape(num_symbols, bits_per_symbol)
        symbol_mean = symbol_abs.mean(axis=1)
        same_symbol_mean = np.repeat(symbol_mean, bits_per_symbol)[:n]
        if fft_size > 0:
            num_ofdm_symbols = max(num_symbols // fft_size, 1)
            re = symbol_mean.reshape(num_ofdm_symbols, fft_size)
            sc_mean = re.mean(axis=0)
            sc_repeat = np.tile(sc_mean, num_ofdm_symbols)
            same_subcarrier_mean = np.repeat(sc_repeat, bits_per_symbol)[:n]

    x = np.stack(
        [
            post,
            abs_post,
            llr_ch,
            abs_llr,
            unsat / max(max_vn_degree, 1),
            flips / max(max_iter, 1),
            idx / max(n - 1, 1),
            same_symbol_mean,
            same_subcarrier_mean,
            (abs_post < 1.0).astype(np.float32),
            (abs_llr < 1.0).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if target_mask is None:
        y = (snapshot.hard ^ trace.true_codeword).astype(np.float32)
    else:
        y = np.asarray(target_mask, dtype=np.float32).reshape(-1)
        if y.size != n:
            raise ValueError(f"target_mask length mismatch: expected {n}, got {y.size}")
    return x, y


def snapshot_training_rows(trace: TraceResult, max_iter: int, max_vn_degree: int, m: int) -> List[Dict]:
    rows: List[Dict] = []
    for snapshot in trace.snapshots:
        feat = snapshot_feature_vector(snapshot, max_iter, max_vn_degree, len(trace.true_codeword), m)
        target = hamming_distance(snapshot.hard, trace.true_codeword) / max(len(trace.true_codeword), 1)
        rows.append(
            {
                "x": feat.tolist(),
                "y": float(target),
                "iter_idx": int(snapshot.iter_idx),
                "syndrome_weight": int(snapshot.syndrome_weight),
            }
        )
    return rows


def teacher_snapshot_training_rows(
    trace: TraceResult,
    teacher_snapshot_idx: int,
    max_iter: int,
    max_vn_degree: int,
    m: int,
) -> List[Dict]:
    """Teacher-aligned snapshot targets.

    Lower targets are better because the selector chooses the argmin. The
    teacher snapshot gets target 0, while non-teacher snapshots get a smooth
    penalty based on distance from the teacher and residual syndrome weight.
    """
    rows: List[Dict] = []
    n_snap = len(trace.snapshots)
    norm = max(n_snap - 1, 1)
    teacher_snapshot_idx = int(np.clip(teacher_snapshot_idx, 0, n_snap - 1))

    for idx, snapshot in enumerate(trace.snapshots):
        feat = snapshot_feature_vector(snapshot, max_iter, max_vn_degree, len(trace.true_codeword), m)
        if idx == teacher_snapshot_idx:
            target = 0.0
        else:
            iter_gap = abs(idx - teacher_snapshot_idx) / norm
            synd_frac = snapshot.syndrome_weight / max(m, 1)
            target = 1.0 + 0.35 * iter_gap + 0.10 * synd_frac
        rows.append(
            {
                "x": feat.tolist(),
                "y": float(target),
                "iter_idx": int(snapshot.iter_idx),
                "syndrome_weight": int(snapshot.syndrome_weight),
                "teacher_snapshot_idx": int(teacher_snapshot_idx + 1),
            }
        )
    return rows
