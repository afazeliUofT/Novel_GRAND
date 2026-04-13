from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import bit_feature_matrix, snapshot_feature_vector
from novel_grand.ldpc.tanner import TannerGraph, mask_to_indices


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(x.size, dtype=np.float32)
    return ranks / float(max(x.size - 1, 1))



def base_shortlist_scores(
    trace: TraceResult,
    snapshot_idx: int,
    graph_struct: TannerGraph,
    *,
    max_iter: int,
    bits_per_symbol: int,
    fft_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-bit base features and a static shortlist score.

    The current TAGS results show that the selector carries useful signal but the
    learned blend stage is weak. This static prior therefore stays conservative:
    it is dominated by LLRs, but includes structural post-BP signals.
    """
    bit_x, _ = bit_feature_matrix(
        trace=trace,
        snapshot_idx=snapshot_idx,
        max_iter=max_iter,
        max_vn_degree=graph_struct.max_vn_degree,
        bits_per_symbol=bits_per_symbol,
        fft_size=fft_size,
    )
    snapshot = trace.snapshots[snapshot_idx]
    inv_abs_post = 1.0 / (np.abs(snapshot.posterior).astype(np.float32) + 1e-3)
    inv_abs_ch = 1.0 / (np.abs(trace.llr_ch).astype(np.float32) + 1e-3)
    unsat = snapshot.unsat_deg.astype(np.float32)
    flips = snapshot.cumulative_flip_count.astype(np.float32)
    scores = (
        0.45 * _rank01(inv_abs_post)
        + 0.20 * _rank01(inv_abs_ch)
        + 0.20 * _rank01(unsat)
        + 0.15 * _rank01(flips)
    ).astype(np.float32)
    return bit_x.astype(np.float32), scores



def greedy_teacher_bit_order(
    trace: TraceResult,
    snapshot_idx: int,
    target_mask: np.ndarray,
    graph_exact: TannerGraph,
) -> List[int]:
    snapshot = trace.snapshots[snapshot_idx]
    bits = np.flatnonzero(np.asarray(target_mask).astype(np.float32) > 0.5).astype(int).tolist()
    if not bits:
        return []

    residual = int(snapshot.syndrome_mask)
    abs_post = np.abs(snapshot.posterior).astype(np.float32)
    unsat = snapshot.unsat_deg.astype(np.float32)
    flips = snapshot.cumulative_flip_count.astype(np.float32)

    order: List[int] = []
    remaining = set(bits)
    while remaining:
        def _key(b: int):
            colmask = graph_exact.col_syndrome_masks[b]
            new_res = residual ^ colmask
            delta = residual.bit_count() - new_res.bit_count()
            return (delta, -float(abs_post[b]), float(unsat[b]), float(flips[b]), -b)

        best = max(remaining, key=_key)
        order.append(int(best))
        residual ^= graph_exact.col_syndrome_masks[int(best)]
        remaining.remove(best)
    return order



def state_feature_vector(
    trace: TraceResult,
    snapshot_idx: int,
    graph_exact: TannerGraph,
    max_iter: int,
    max_vn_degree: int,
    selected_mask: int,
    residual_mask: int,
) -> np.ndarray:
    snapshot = trace.snapshots[snapshot_idx]
    base = snapshot_feature_vector(snapshot, max_iter, max_vn_degree, graph_exact.n, graph_exact.m)
    sel_bits = mask_to_indices(int(selected_mask))
    if sel_bits:
        abs_post = np.abs(snapshot.posterior[sel_bits]).astype(np.float32)
        unsat = snapshot.unsat_deg[sel_bits].astype(np.float32)
        flips = snapshot.cumulative_flip_count[sel_bits].astype(np.float32)
        colw = np.asarray([len(graph_exact.bit_to_checks[b]) for b in sel_bits], dtype=np.float32)
        sel_abs_mean = float(abs_post.mean())
        sel_abs_min = float(abs_post.min(initial=0.0))
        sel_unsat_mean = float(unsat.mean()) / max(max_vn_degree, 1)
        sel_flips_mean = float(flips.mean()) / max(max_iter, 1)
        sel_colw_mean = float(colw.mean()) / max(graph_exact.m, 1)
    else:
        sel_abs_mean = 0.0
        sel_abs_min = 0.0
        sel_unsat_mean = 0.0
        sel_flips_mean = 0.0
        sel_colw_mean = 0.0

    extra = np.array(
        [
            residual_mask.bit_count() / max(graph_exact.m, 1),
            len(sel_bits) / max(graph_exact.n, 1),
            1.0 if sel_bits else 0.0,
            sel_abs_mean,
            sel_abs_min,
            sel_unsat_mean,
            sel_flips_mean,
            sel_colw_mean,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra], axis=0).astype(np.float32)



def candidate_feature_matrix(
    trace: TraceResult,
    snapshot_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    *,
    max_iter: int,
    bits_per_symbol: int,
    fft_size: int,
    selected_mask: int,
    residual_mask: int,
    candidate_bits: Sequence[int],
) -> np.ndarray:
    bit_x, _ = base_shortlist_scores(
        trace,
        snapshot_idx,
        graph_struct,
        max_iter=max_iter,
        bits_per_symbol=bits_per_symbol,
        fft_size=fft_size,
    )
    snapshot = trace.snapshots[snapshot_idx]
    sel_bits = mask_to_indices(int(selected_mask))
    if sel_bits:
        sel_abs_mean = float(np.abs(snapshot.posterior[sel_bits]).mean())
        sel_unsat_mean = float(snapshot.unsat_deg[sel_bits].mean()) / max(graph_struct.max_vn_degree, 1)
        sel_flips_mean = float(snapshot.cumulative_flip_count[sel_bits].mean()) / max(max_iter, 1)
    else:
        sel_abs_mean = 0.0
        sel_unsat_mean = 0.0
        sel_flips_mean = 0.0

    resid_w = residual_mask.bit_count()
    rows = []
    for b in candidate_bits:
        b = int(b)
        colmask = graph_exact.col_syndrome_masks[b]
        new_res = residual_mask ^ colmask
        new_w = new_res.bit_count()
        delta = resid_w - new_w
        colw = len(graph_exact.bit_to_checks[b])
        touch = (residual_mask & colmask).bit_count()
        rows.append(
            np.concatenate(
                [
                    bit_x[b],
                    np.array(
                        [
                            resid_w / max(graph_exact.m, 1),
                            new_w / max(graph_exact.m, 1),
                            delta / max(graph_exact.m, 1),
                            touch / max(colw, 1),
                            len(sel_bits) / max(graph_exact.n, 1),
                            sel_abs_mean,
                            sel_unsat_mean,
                            sel_flips_mean,
                        ],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        )
    return np.asarray(rows, dtype=np.float32)



def build_policy_value_training_rows(
    trace: TraceResult,
    snapshot_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    *,
    max_iter: int,
    bits_per_symbol: int,
    fft_size: int,
    target_mask: np.ndarray | None,
    shortlist_topk_bits: int,
    neg_to_pos_ratio: int,
    value_negatives_per_prefix: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build policy-prior and state-value training rows.

    Policy rows teach a next-action prior over a conservative shortlist.
    Value rows teach whether a partial state lies on a successful teacher path.
    """
    bit_x, shortlist_scores = base_shortlist_scores(
        trace,
        snapshot_idx,
        graph_struct,
        max_iter=max_iter,
        bits_per_symbol=bits_per_symbol,
        fft_size=fft_size,
    )
    shortlist = np.argsort(shortlist_scores)[::-1][: int(shortlist_topk_bits)].astype(int).tolist()

    action_x_rows: List[np.ndarray] = []
    action_y_rows: List[np.ndarray] = []
    value_x_rows: List[np.ndarray] = []
    value_y_rows: List[np.ndarray] = []

    snapshot = trace.snapshots[snapshot_idx]
    if target_mask is None or np.count_nonzero(target_mask) == 0:
        # Hard negative from an unresolved frame.
        value_x_rows.append(
            state_feature_vector(
                trace,
                snapshot_idx,
                graph_exact,
                max_iter,
                graph_struct.max_vn_degree,
                selected_mask=0,
                residual_mask=int(snapshot.syndrome_mask),
            )
        )
        value_y_rows.append(np.array([0.0], dtype=np.float32))
        return (
            np.zeros((0, bit_x.shape[1] + 8), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.asarray(value_x_rows, dtype=np.float32),
            np.asarray(value_y_rows, dtype=np.float32).reshape(-1),
        )

    teacher_order = greedy_teacher_bit_order(trace, snapshot_idx, target_mask, graph_exact)
    shortlist = sorted(set(shortlist).union(teacher_order))

    selected_mask = 0
    residual_mask = int(snapshot.syndrome_mask)
    remaining = list(teacher_order)

    for step, next_bit in enumerate(teacher_order):
        # Positive value row for the current prefix.
        value_x_rows.append(
            state_feature_vector(
                trace,
                snapshot_idx,
                graph_exact,
                max_iter,
                graph_struct.max_vn_degree,
                selected_mask=selected_mask,
                residual_mask=residual_mask,
            )
        )
        value_y_rows.append(np.array([1.0], dtype=np.float32))

        cand = [b for b in shortlist if ((selected_mask >> int(b)) & 1) == 0]
        if next_bit not in cand:
            cand = [int(next_bit)] + cand
        x_cand = candidate_feature_matrix(
            trace,
            snapshot_idx,
            graph_exact,
            graph_struct,
            max_iter=max_iter,
            bits_per_symbol=bits_per_symbol,
            fft_size=fft_size,
            selected_mask=selected_mask,
            residual_mask=residual_mask,
            candidate_bits=cand,
        )
        y_cand = np.zeros(len(cand), dtype=np.float32)
        y_cand[cand.index(int(next_bit))] = 1.0
        pos_idx = np.flatnonzero(y_cand > 0.5)
        neg_idx = np.flatnonzero(y_cand <= 0.5)
        keep_neg = min(len(neg_idx), max(1, neg_to_pos_ratio) * len(pos_idx))
        if keep_neg > 0:
            keep_neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False)
            keep = np.concatenate([pos_idx, keep_neg_idx])
        else:
            keep = pos_idx
        action_x_rows.append(x_cand[keep])
        action_y_rows.append(y_cand[keep])

        # Hard negatives at the same depth.
        bad_pool = [b for b in cand if b != int(next_bit)]
        if bad_pool:
            n_bad = min(len(bad_pool), max(1, value_negatives_per_prefix))
            bad_bits = rng.choice(np.asarray(bad_pool, dtype=int), size=n_bad, replace=False)
            for bad in np.asarray(bad_bits).reshape(-1).tolist():
                bad = int(bad)
                bad_mask = selected_mask ^ (1 << bad)
                bad_res = residual_mask ^ graph_exact.col_syndrome_masks[bad]
                value_x_rows.append(
                    state_feature_vector(
                        trace,
                        snapshot_idx,
                        graph_exact,
                        max_iter,
                        graph_struct.max_vn_degree,
                        selected_mask=bad_mask,
                        residual_mask=bad_res,
                    )
                )
                value_y_rows.append(np.array([0.0], dtype=np.float32))

        selected_mask ^= 1 << int(next_bit)
        residual_mask ^= graph_exact.col_syndrome_masks[int(next_bit)]
        if remaining:
            remaining.pop(0)

    # Terminal successful state.
    value_x_rows.append(
        state_feature_vector(
            trace,
            snapshot_idx,
            graph_exact,
            max_iter,
            graph_struct.max_vn_degree,
            selected_mask=selected_mask,
            residual_mask=residual_mask,
        )
    )
    value_y_rows.append(np.array([1.0], dtype=np.float32))

    action_x = np.concatenate(action_x_rows, axis=0).astype(np.float32) if action_x_rows else np.zeros((0, bit_x.shape[1] + 8), dtype=np.float32)
    action_y = np.concatenate(action_y_rows, axis=0).astype(np.float32) if action_y_rows else np.zeros((0,), dtype=np.float32)
    value_x = np.asarray(value_x_rows, dtype=np.float32)
    value_y = np.asarray(value_y_rows, dtype=np.float32).reshape(-1)
    return action_x, action_y, value_x, value_y
