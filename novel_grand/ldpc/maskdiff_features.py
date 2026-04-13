from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import snapshot_feature_vector
from novel_grand.ldpc.pv_features import base_shortlist_scores
from novel_grand.ldpc.tanner import TannerGraph


MASK_VALUE = -1


def shortlist_with_required_bits(
    trace: TraceResult,
    snapshot_idx: int,
    graph_struct: TannerGraph,
    cfg,
    *,
    topk_bits: int,
    required_bits: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return base bit features, shortlist indices, and conservative shortlist scores.

    The shortlist is dominated by LLR-driven risk but can be *forced* to contain
    teacher bits during training. This is important because the current rescue
    regime is rare-event and exact patterns may otherwise fall just outside the
    shortlist.
    """
    bit_x, base_scores = base_shortlist_scores(
        trace,
        snapshot_idx,
        graph_struct,
        max_iter=int(cfg["legacy_ldpc"]["num_iter"]),
        bits_per_symbol=int(cfg["nr"]["bits_per_symbol"]),
        fft_size=int(cfg["nr"]["fft_size"]),
    )
    order = np.argsort(base_scores)[::-1].astype(int).tolist()
    shortlist = order[: int(topk_bits)]

    if required_bits:
        keep = [int(b) for b in required_bits if 0 <= int(b) < bit_x.shape[0]]
        for b in keep:
            if b in shortlist:
                continue
            if shortlist:
                shortlist.pop(-1)
            shortlist.insert(0, b)
        # Stable de-duplication while keeping length fixed.
        uniq: List[int] = []
        seen = set()
        for b in shortlist + order:
            b = int(b)
            if b in seen:
                continue
            seen.add(b)
            uniq.append(b)
            if len(uniq) >= int(topk_bits):
                break
        shortlist = uniq

    return bit_x.astype(np.float32), np.asarray(shortlist, dtype=np.int64), base_scores.astype(np.float32)



def token_feature_matrix(
    base_bit_x_short: np.ndarray,
    state: np.ndarray,
    *,
    step_frac: float,
    consensus_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Compose per-token features for the masked diffusion denoiser.

    state values:
      -1 -> masked / unknown
       0 -> revealed no-flip
       1 -> revealed flip
    """
    state = np.asarray(state, dtype=np.int8).reshape(-1)
    k = state.size
    masked = (state < 0).astype(np.float32)
    known_zero = (state == 0).astype(np.float32)
    known_one = (state > 0).astype(np.float32)
    if consensus_bias is None:
        consensus_bias = np.zeros(k, dtype=np.float32)
    else:
        consensus_bias = np.asarray(consensus_bias, dtype=np.float32).reshape(-1)
        if consensus_bias.size != k:
            raise ValueError(f"consensus_bias length mismatch: expected {k}, got {consensus_bias.size}")
    step_col = np.full((k,), float(step_frac), dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(base_bit_x_short, dtype=np.float32),
            np.stack([masked, known_zero, known_one], axis=1),
            step_col[:, None],
            consensus_bias[:, None],
        ],
        axis=1,
    ).astype(np.float32)



def global_feature_vector(
    trace: TraceResult,
    snapshot_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    cfg,
    *,
    state: np.ndarray,
    residual_mask: int,
    step_frac: float,
) -> np.ndarray:
    snapshot = trace.snapshots[snapshot_idx]
    base = snapshot_feature_vector(
        snapshot,
        max_iter=int(cfg["legacy_ldpc"]["num_iter"]),
        max_vn_degree=graph_struct.max_vn_degree,
        n=graph_exact.n,
        m=graph_exact.m,
    )
    state = np.asarray(state, dtype=np.int8).reshape(-1)
    masked_frac = float(np.mean(state < 0))
    known_frac = float(np.mean(state >= 0))
    flip_frac = float(np.mean(state > 0))
    extra = np.array(
        [
            masked_frac,
            known_frac,
            flip_frac,
            int(residual_mask).bit_count() / max(graph_exact.m, 1),
            float(step_frac),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra], axis=0).astype(np.float32)



def mask_from_shortlist(shortlist: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    shortlist = np.asarray(shortlist, dtype=np.int64).reshape(-1)
    target_mask = np.asarray(target_mask, dtype=np.float32).reshape(-1)
    return target_mask[shortlist].astype(np.float32)



def build_maskdiff_training_rows(
    trace: TraceResult,
    snapshot_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    cfg,
    *,
    target_mask: np.ndarray | None,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build masked-diffusion denoising rows for post-guard failures.

    Returns:
      token_x: [N, K, F_tok]
      global_x: [N, F_glb]
      y:       [N, K]
      loss_mask:[N, K]
    """
    topk_bits = int(cfg["grand"].get("mdiff_shortlist_bits", 64))
    n_corruptions = int(cfg["training"].get("maskdiff_corruptions_per_frame", 6))

    if target_mask is None:
        return (
            np.zeros((0, topk_bits, 16), dtype=np.float32),
            np.zeros((0, 17), dtype=np.float32),
            np.zeros((0, topk_bits), dtype=np.float32),
            np.zeros((0, topk_bits), dtype=np.float32),
        )

    required_bits = np.flatnonzero(np.asarray(target_mask).astype(np.float32) > 0.5).astype(int).tolist()
    bit_x, shortlist, _ = shortlist_with_required_bits(
        trace,
        snapshot_idx,
        graph_struct,
        cfg,
        topk_bits=topk_bits,
        required_bits=required_bits,
    )
    base_short = bit_x[shortlist]
    y_short = mask_from_shortlist(shortlist, np.asarray(target_mask, dtype=np.float32))
    k = int(shortlist.size)

    token_rows: List[np.ndarray] = []
    global_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []

    pos_idx = np.flatnonzero(y_short > 0.5).astype(int)
    neg_idx = np.flatnonzero(y_short <= 0.5).astype(int)

    reveal_fracs = [0.0, 0.10, 0.20, 0.35, 0.50]

    for i in range(max(n_corruptions, 1)):
        reveal_frac = float(reveal_fracs[i % len(reveal_fracs)])
        n_reveal = int(round(reveal_frac * k))

        state = np.full((k,), MASK_VALUE, dtype=np.int8)
        if n_reveal > 0:
            n_reveal_pos = min(int(round(0.5 * n_reveal)), int(pos_idx.size))
            n_reveal_neg = min(n_reveal - n_reveal_pos, int(neg_idx.size))
            picked = []
            if n_reveal_pos > 0:
                picked.extend(rng.choice(pos_idx, size=n_reveal_pos, replace=False).astype(int).tolist())
            if n_reveal_neg > 0:
                picked.extend(rng.choice(neg_idx, size=n_reveal_neg, replace=False).astype(int).tolist())
            # Top up if either class was too small.
            if len(picked) < n_reveal:
                remaining = [j for j in range(k) if j not in set(picked)]
                if remaining:
                    extra = rng.choice(np.asarray(remaining, dtype=np.int64), size=min(len(remaining), n_reveal - len(picked)), replace=False)
                    picked.extend(np.asarray(extra, dtype=int).tolist())
            if picked:
                picked = sorted(set(int(j) for j in picked))
                state[picked] = y_short[picked].astype(np.int8)

        loss_mask = (state < 0).astype(np.float32)
        if loss_mask.sum() <= 0:
            # Leave at least one position masked.
            j = int(rng.integers(0, k))
            state[j] = MASK_VALUE
            loss_mask = (state < 0).astype(np.float32)

        token_rows.append(token_feature_matrix(base_short, state, step_frac=reveal_frac))
        global_rows.append(
            global_feature_vector(
                trace,
                snapshot_idx,
                graph_exact,
                graph_struct,
                cfg,
                state=state,
                residual_mask=int(trace.snapshots[snapshot_idx].syndrome_mask),
                step_frac=reveal_frac,
            )
        )
        y_rows.append(y_short.astype(np.float32))
        mask_rows.append(loss_mask.astype(np.float32))

    return (
        np.asarray(token_rows, dtype=np.float32),
        np.asarray(global_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=np.float32),
    )
