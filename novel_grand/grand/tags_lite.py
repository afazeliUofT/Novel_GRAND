from __future__ import annotations

from typing import Dict, List

import numpy as np

from novel_grand.grand.search import Primitive, search_exact_syndrome
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import bit_feature_matrix, snapshot_feature_vector
from novel_grand.ldpc.tanner import TannerGraph
from novel_grand.models.training import BitRankerModel, SnapshotSelectorModel


def _build_group_primitives(
    graph_struct: TannerGraph,
    graph_exact: TannerGraph,
    scores: np.ndarray,
    flips: np.ndarray,
    unsat: np.ndarray,
    syndrome_mask: int,
    cfg,
) -> List[Primitive]:
    gcfg = cfg["grand"]
    nr_cfg = cfg["nr"]
    topk = int(gcfg["topk_bits"])
    max_groups = int(gcfg["max_group_count"])
    group_bonus_log_size = float(gcfg["group_bonus_log_size"])

    top_bits = np.argsort(scores)[::-1][:topk].astype(int).tolist()
    groups = []
    groups.extend(graph_struct.unsatisfied_components(syndrome_mask, max_groups=max_groups // 3 or 1))
    groups.extend(graph_struct.top_unsatisfied_check_groups(syndrome_mask, max_groups=max_groups // 3 or 1))
    groups.extend(graph_struct.contiguous_symbol_groups(int(nr_cfg["bits_per_symbol"]), top_bits, max_groups=max_groups // 6 or 1))
    groups.extend(graph_struct.subcarrier_groups(int(nr_cfg["bits_per_symbol"]), int(nr_cfg["fft_size"]), top_bits, max_groups=max_groups // 6 or 1))

    out: List[Primitive] = []
    seen_masks = set()
    for grp in groups:
        bits = sorted(set(int(b) for b in grp.bit_indices))
        if len(bits) <= 1:
            continue
        bit_mask = 0
        syn_mask = 0
        for b in bits:
            bit_mask ^= (1 << b)
            syn_mask ^= graph_exact.col_syndrome_masks[b]
        if bit_mask in seen_masks:
            continue
        seen_masks.add(bit_mask)
        base_score = float(scores[bits].mean())
        flip_bonus = float(np.log1p(flips[bits].mean()))
        unsat_bonus = float(unsat[bits].mean()) / max(graph_struct.max_vn_degree, 1)
        size_bonus = group_bonus_log_size * float(np.log1p(len(bits)))
        out.append(
            Primitive(
                name=grp.name,
                kind="group",
                bit_indices=bits,
                bit_mask=bit_mask,
                syn_mask=syn_mask,
                score=base_score + 0.1 * flip_bonus + 0.2 * unsat_bonus + size_bonus,
            )
        )
        if len(out) >= max_groups:
            break
    return out


def run_tags_grand_lite(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    bit_model: BitRankerModel,
    cfg,
) -> Dict:
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    n = graph_exact.n
    m = graph_exact.m
    snap_feats = np.stack(
        [
            snapshot_feature_vector(s, max_iter, graph_struct.max_vn_degree, n, m)
            for s in trace.snapshots
        ],
        axis=0,
    )
    snap_idx = snapshot_model.select(snap_feats)
    snapshot = trace.snapshots[snap_idx]

    bit_x, _ = bit_feature_matrix(
        trace=trace,
        snapshot_idx=snap_idx,
        max_iter=max_iter,
        max_vn_degree=graph_struct.max_vn_degree,
        bits_per_symbol=int(cfg["nr"]["bits_per_symbol"]),
        fft_size=int(cfg["nr"]["fft_size"]),
    )
    bit_prob = bit_model.predict_prob(bit_x)
    bit_scores = bit_prob.astype(np.float32)

    top_bits = np.argsort(bit_scores)[::-1][: int(cfg["grand"]["topk_bits"])]
    primitives: List[Primitive] = []
    for j in top_bits:
        j = int(j)
        primitives.append(
            Primitive(
                name=f"bit_{j}",
                kind="bit",
                bit_indices=[j],
                bit_mask=(1 << j),
                syn_mask=graph_exact.col_syndrome_masks[j],
                score=float(bit_scores[j]),
            )
        )

    group_prims = _build_group_primitives(
        graph_struct=graph_struct,
        graph_exact=graph_exact,
        scores=bit_scores,
        flips=snapshot.cumulative_flip_count.astype(np.float32),
        unsat=snapshot.unsat_deg.astype(np.float32),
        syndrome_mask=snapshot.structure_syndrome_mask,
        cfg=cfg,
    )
    primitives.extend(group_prims)
    primitives = sorted(primitives, key=lambda p: p.score, reverse=True)

    res = search_exact_syndrome(
        n=graph_exact.n,
        hard_bits=snapshot.hard,
        syndrome_mask=snapshot.syndrome_mask,
        primitives=primitives,
        query_cap=int(cfg["grand"]["query_cap"]),
        max_primitives_in_pattern=int(cfg["grand"]["max_primitives_in_pattern"]),
        expand_width=int(cfg["grand"]["search_expand_width"]),
        overlap_penalty=float(cfg["grand"]["overlap_penalty"]),
    )
    valid = graph_exact.syndrome_mask(res.corrected_bits) == 0
    exact = bool(np.array_equal(res.corrected_bits.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    return {
        "decoder": "tags_grand_lite",
        "selected_snapshot": snap_idx + 1,
        "selected_syndrome_weight": int(snapshot.syndrome_weight),
        "queries": int(res.queries),
        "frontier_peak": int(res.frontier_peak),
        "pattern_weight": int(res.pattern_mask.bit_count()),
        "success_exact": exact,
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": ",".join(res.selected_primitive_kinds),
        "primitive_sizes": ",".join(map(str, res.selected_primitive_sizes)),
    }
