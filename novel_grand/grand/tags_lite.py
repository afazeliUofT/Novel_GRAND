from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from novel_grand.grand.baselines import llr_risk, run_baseline
from novel_grand.grand.search import Primitive, search_exact_syndrome
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import bit_feature_matrix, snapshot_feature_vector
from novel_grand.ldpc.tanner import TannerGraph
from novel_grand.models.training import BitRankerModel, SnapshotSelectorModel


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(x.size, dtype=np.float32)
    return ranks / float(max(x.size - 1, 1))



def _individual_primitives(graph_exact: TannerGraph, scores: np.ndarray, topk_bits: int) -> List[Primitive]:
    idx = np.argsort(scores)[::-1][: int(topk_bits)]
    out: List[Primitive] = []
    for j in idx:
        j = int(j)
        out.append(
            Primitive(
                name=f"bit_{j}",
                kind="bit",
                bit_indices=[j],
                bit_mask=(1 << j),
                syn_mask=graph_exact.col_syndrome_masks[j],
                score=float(scores[j]),
            )
        )
    return out



def _build_group_primitives(
    graph_struct: TannerGraph,
    graph_exact: TannerGraph,
    scores: np.ndarray,
    flips: np.ndarray,
    unsat: np.ndarray,
    syndrome_mask: int,
    cfg,
    *,
    topk_bits: int,
    max_groups: int,
) -> List[Primitive]:
    nr_cfg = cfg["nr"]
    group_bonus_log_size = float(cfg["grand"].get("group_bonus_log_size", 0.10))

    top_bits = np.argsort(scores)[::-1][: int(topk_bits)].astype(int).tolist()
    groups = []
    groups.extend(graph_struct.unsatisfied_components(syndrome_mask, max_groups=max_groups // 3 or 1))
    groups.extend(graph_struct.top_unsatisfied_check_groups(syndrome_mask, max_groups=max_groups // 3 or 1))
    groups.extend(
        graph_struct.contiguous_symbol_groups(
            int(nr_cfg["bits_per_symbol"]),
            top_bits,
            max_groups=max_groups // 6 or 1,
        )
    )
    groups.extend(
        graph_struct.subcarrier_groups(
            int(nr_cfg["bits_per_symbol"]),
            int(nr_cfg["fft_size"]),
            top_bits,
            max_groups=max_groups // 6 or 1,
        )
    )

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
                score=base_score + 0.10 * flip_bonus + 0.18 * unsat_bonus + size_bonus,
            )
        )
        if len(out) >= max_groups:
            break
    return out



def _snapshot_candidates(
    trace: TraceResult,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    max_iter: int,
    n: int,
    m: int,
) -> Tuple[int, List[int], np.ndarray]:
    snap_feats = np.stack(
        [snapshot_feature_vector(s, max_iter, graph_struct.max_vn_degree, n, m) for s in trace.snapshots],
        axis=0,
    )
    pred = snapshot_model.predict(snap_feats).reshape(-1)
    n_snap = len(trace.snapshots)
    final_idx = n_snap - 1
    synd = np.asarray([s.syndrome_weight for s in trace.snapshots], dtype=np.float32)
    model_order = list(np.argsort(pred))
    min_synd_idx = int(np.argmin(synd))

    candidates: List[int] = []

    def _add(idx: int) -> None:
        if 0 <= idx < n_snap and idx not in candidates:
            candidates.append(int(idx))

    # Teacher-aligned selector: trust the model first, then provide conservative
    # structural fallbacks in case the chosen snapshot is poor for a given frame.
    _add(int(model_order[0]))
    if len(model_order) > 1:
        _add(int(model_order[1]))
    _add(min_synd_idx)
    _add(final_idx)

    chosen = candidates[0]
    ordered = [chosen] + [i for i in candidates if i != chosen]
    return chosen, ordered, pred



def _bit_scores_for_snapshot(
    trace: TraceResult,
    snap_idx: int,
    graph_struct: TannerGraph,
    bit_model: BitRankerModel,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    bit_x, _ = bit_feature_matrix(
        trace=trace,
        snapshot_idx=snap_idx,
        max_iter=max_iter,
        max_vn_degree=graph_struct.max_vn_degree,
        bits_per_symbol=int(cfg["nr"]["bits_per_symbol"]),
        fft_size=int(cfg["nr"]["fft_size"]),
    )
    bit_prob = bit_model.predict_prob(bit_x).astype(np.float32)
    snapshot = trace.snapshots[snap_idx]

    inv_abs_post = 1.0 / (np.abs(snapshot.posterior).astype(np.float32) + 1e-3)
    inv_abs_ch = 1.0 / (np.abs(trace.llr_ch).astype(np.float32) + 1e-3)
    unsat = snapshot.unsat_deg.astype(np.float32)
    flips = snapshot.cumulative_flip_count.astype(np.float32)

    # After teacher-aligned training, the learned probability should dominate.
    scores = (
        0.62 * _rank01(bit_prob)
        + 0.20 * _rank01(inv_abs_post)
        + 0.10 * _rank01(inv_abs_ch)
        + 0.08 * _rank01(unsat + 0.20 * flips)
    ).astype(np.float32)
    return bit_prob, scores



def _search_from_scores(
    trace: TraceResult,
    snap_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    scores: np.ndarray,
    cfg,
    query_cap: int,
    *,
    stage_tag: str,
    include_groups: bool,
    topk_bits: int,
    expand_width: int,
    max_prims: int,
    max_groups: int,
) -> Dict:
    snapshot = trace.snapshots[snap_idx]
    primitives: List[Primitive] = _individual_primitives(graph_exact, scores, topk_bits=topk_bits)
    if include_groups:
        primitives.extend(
            _build_group_primitives(
                graph_struct=graph_struct,
                graph_exact=graph_exact,
                scores=scores,
                flips=snapshot.cumulative_flip_count.astype(np.float32),
                unsat=snapshot.unsat_deg.astype(np.float32),
                syndrome_mask=snapshot.structure_syndrome_mask,
                cfg=cfg,
                topk_bits=topk_bits,
                max_groups=max_groups,
            )
        )
    primitives = sorted(primitives, key=lambda p: p.score, reverse=True)

    res = search_exact_syndrome(
        n=graph_exact.n,
        hard_bits=snapshot.hard,
        syndrome_mask=snapshot.syndrome_mask,
        primitives=primitives,
        query_cap=int(query_cap),
        max_primitives_in_pattern=int(max_prims),
        expand_width=int(expand_width),
        overlap_penalty=float(cfg["grand"].get("overlap_penalty", 0.35)),
    )
    valid = graph_exact.syndrome_mask(res.corrected_bits) == 0
    exact = bool(np.array_equal(res.corrected_bits.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    selected_kinds = ",".join(res.selected_primitive_kinds) if res.selected_primitive_kinds else ""
    selected_sizes = ",".join(map(str, res.selected_primitive_sizes))
    return {
        "selected_snapshot": snap_idx + 1,
        "selected_syndrome_weight": int(snapshot.syndrome_weight),
        "queries": int(res.queries),
        "query_budget": int(query_cap),
        "frontier_peak": int(res.frontier_peak),
        "pattern_weight": int(res.pattern_mask.bit_count()),
        "success_exact": bool(exact),
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": stage_tag + (("|" + selected_kinds) if selected_kinds else ""),
        "primitive_sizes": selected_sizes,
    }



def _search_stage_llr(
    trace: TraceResult,
    snap_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    cfg,
    query_cap: int,
    *,
    stage_tag: str,
    topk_bits: int,
    expand_width: int,
    max_prims: int,
) -> Dict:
    del graph_struct
    scores = llr_risk(trace.snapshots[snap_idx])
    out = _search_from_scores(
        trace=trace,
        snap_idx=snap_idx,
        graph_exact=graph_exact,
        graph_struct=graph_exact,
        scores=scores,
        cfg=cfg,
        query_cap=query_cap,
        stage_tag=stage_tag,
        include_groups=False,
        topk_bits=topk_bits,
        expand_width=expand_width,
        max_prims=max_prims,
        max_groups=0,
    )
    out["decoder"] = "tags_grand_lite"
    return out



def _search_stage_blend(
    trace: TraceResult,
    snap_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    bit_model: BitRankerModel,
    cfg,
    query_cap: int,
    *,
    stage_tag: str,
    include_groups: bool,
    topk_bits: int,
    expand_width: int,
    max_prims: int,
    max_groups: int,
) -> Dict:
    _, scores = _bit_scores_for_snapshot(trace, snap_idx, graph_struct, bit_model, cfg)
    out = _search_from_scores(
        trace=trace,
        snap_idx=snap_idx,
        graph_exact=graph_exact,
        graph_struct=graph_struct,
        scores=scores,
        cfg=cfg,
        query_cap=query_cap,
        stage_tag=stage_tag,
        include_groups=include_groups,
        topk_bits=topk_bits,
        expand_width=expand_width,
        max_prims=max_prims,
        max_groups=max_groups,
    )
    out["decoder"] = "tags_grand_lite"
    return out



def _merge_results(primary: Dict, secondary: Dict) -> Dict:
    out = dict(secondary)
    out["queries"] = int(primary.get("queries", 0)) + int(secondary.get("queries", 0))
    out["query_budget"] = int(primary.get("query_budget", 0)) + int(secondary.get("query_budget", 0))
    out["frontier_peak"] = int(max(primary.get("frontier_peak", 0), secondary.get("frontier_peak", 0)))
    pk1 = str(primary.get("primitive_kinds", ""))
    pk2 = str(secondary.get("primitive_kinds", ""))
    out["primitive_kinds"] = "|".join([x for x in [pk1, pk2] if x])
    ps1 = str(primary.get("primitive_sizes", ""))
    ps2 = str(secondary.get("primitive_sizes", ""))
    out["primitive_sizes"] = "|".join([x for x in [ps1, ps2] if x])
    return out



def run_selector_llr_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    cfg,
) -> Dict:
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    chosen_idx, _, _ = _snapshot_candidates(
        trace=trace,
        graph_struct=graph_struct,
        snapshot_model=snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
    )
    gcfg = cfg["grand"]
    out = _search_stage_llr(
        trace=trace,
        snap_idx=chosen_idx,
        graph_exact=graph_exact,
        graph_struct=graph_struct,
        cfg=cfg,
        query_cap=int(gcfg["query_cap"]),
        stage_tag="selector_llr",
        topk_bits=max(int(gcfg.get("selector_llr_topk_bits", max(128, int(gcfg.get("topk_bits", 96))))), 96),
        expand_width=max(int(gcfg.get("selector_llr_expand_width", max(24, int(gcfg.get("search_expand_width", 10))))), 12),
        max_prims=max(int(gcfg.get("selector_llr_max_primitives_in_pattern", max(12, int(gcfg.get("max_primitives_in_pattern", 10))))), 10),
    )
    out["decoder"] = "selector_llr_grand"
    return out



def run_selector_blend_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    bit_model: BitRankerModel,
    cfg,
) -> Dict:
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    chosen_idx, _, _ = _snapshot_candidates(
        trace=trace,
        graph_struct=graph_struct,
        snapshot_model=snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
    )
    gcfg = cfg["grand"]
    out = _search_stage_blend(
        trace=trace,
        snap_idx=chosen_idx,
        graph_exact=graph_exact,
        graph_struct=graph_struct,
        bit_model=bit_model,
        cfg=cfg,
        query_cap=int(gcfg["query_cap"]),
        stage_tag="selector_blend",
        include_groups=False,
        topk_bits=max(int(gcfg.get("selector_blend_topk_bits", max(128, int(gcfg.get("topk_bits", 96))))), 96),
        expand_width=max(int(gcfg.get("selector_blend_expand_width", max(24, int(gcfg.get("search_expand_width", 10))))), 12),
        max_prims=max(int(gcfg.get("selector_blend_max_primitives_in_pattern", max(12, int(gcfg.get("max_primitives_in_pattern", 10))))), 10),
        max_groups=max(int(gcfg.get("selector_blend_max_group_count", max(32, int(gcfg.get("max_group_count", 20))))), 16),
    )
    out["decoder"] = "selector_blend_grand"
    return out



def run_tags_grand_lite(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    bit_model: BitRankerModel,
    cfg,
) -> Dict:
    """Hybrid rescue with teacher-aligned snapshot selection.

    Stage 1: strongest practical non-AI guard on the final snapshot.
    Stage 2: AI-selected snapshot with LLR ordering (AI-aided because the
             snapshot comes from the learned selector).
    Stage 3: learned blend search on the same selected snapshot.
    Stage 4: conservative best-syndrome fallback.
    """
    base_cfg = copy.deepcopy(cfg)
    q_main = int(base_cfg["grand"]["query_cap"])
    rescue_bonus_cap = int(base_cfg["grand"].get("rescue_bonus_cap", max(2000, q_main // 2)))
    fallback_bonus_cap = int(base_cfg["grand"].get("fallback_bonus_cap", max(1000, q_main // 4)))

    gcfg = base_cfg["grand"]
    ai_selector_fraction = float(gcfg.get("ai_selector_fraction", 0.70))
    ai_selector_cap = int(gcfg.get("ai_selector_cap", round(ai_selector_fraction * rescue_bonus_cap)))
    ai_selector_cap = max(0, min(ai_selector_cap, rescue_bonus_cap))
    ai_blend_cap = max(0, rescue_bonus_cap - ai_selector_cap)

    guard_base = run_baseline("final_llr_grand", trace, graph_exact, base_cfg)
    guard = dict(guard_base)
    guard["decoder"] = "tags_grand_lite"
    guard["query_budget"] = int(q_main + rescue_bonus_cap + fallback_bonus_cap)
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard_base.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    chosen_idx, _, _ = _snapshot_candidates(
        trace=trace,
        graph_struct=graph_struct,
        snapshot_model=snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
    )

    accumulated = dict(guard)

    if ai_selector_cap > 0:
        ai_selector = _search_stage_llr(
            trace=trace,
            snap_idx=chosen_idx,
            graph_exact=graph_exact,
            graph_struct=graph_struct,
            cfg=cfg,
            query_cap=ai_selector_cap,
            stage_tag="ai_selector_llr",
            topk_bits=max(int(gcfg.get("ai_selector_topk_bits", max(144, int(gcfg.get("topk_bits", 96))))), 96),
            expand_width=max(int(gcfg.get("ai_selector_expand_width", 24)), 12),
            max_prims=max(int(gcfg.get("ai_selector_max_primitives_in_pattern", 12)), 10),
        )
        accumulated = _merge_results(accumulated, ai_selector)
        if ai_selector.get("valid_codeword", False):
            accumulated["query_budget"] = int(q_main + rescue_bonus_cap + fallback_bonus_cap)
            return accumulated

    if ai_blend_cap > 0:
        ai_blend = _search_stage_blend(
            trace=trace,
            snap_idx=chosen_idx,
            graph_exact=graph_exact,
            graph_struct=graph_struct,
            bit_model=bit_model,
            cfg=cfg,
            query_cap=ai_blend_cap,
            stage_tag="ai_blend",
            include_groups=True,
            topk_bits=max(int(gcfg.get("ai_blend_topk_bits", max(160, int(gcfg.get("topk_bits", 96))))), 128),
            expand_width=max(int(gcfg.get("ai_blend_expand_width", 48)), 16),
            max_prims=max(int(gcfg.get("ai_blend_max_primitives_in_pattern", 12)), 10),
            max_groups=max(int(gcfg.get("ai_blend_max_group_count", max(32, int(gcfg.get("max_group_count", 20))))), 16),
        )
        accumulated = _merge_results(accumulated, ai_blend)
        if ai_blend.get("valid_codeword", False):
            accumulated["query_budget"] = int(q_main + rescue_bonus_cap + fallback_bonus_cap)
            return accumulated

    fb_cfg = copy.deepcopy(cfg)
    fb_cfg["grand"] = dict(cfg["grand"])
    fb_cfg["grand"]["query_cap"] = int(fallback_bonus_cap)
    fallback_base = run_baseline("best_syndrome_llr_grand", trace, graph_exact, fb_cfg)
    fallback = dict(fallback_base)
    fallback["decoder"] = "tags_grand_lite"
    fallback["query_budget"] = int(fb_cfg["grand"]["query_cap"])
    fallback["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr", str(fallback_base.get("primitive_kinds", ""))] if x])
    accumulated = _merge_results(accumulated, fallback)
    accumulated["query_budget"] = int(q_main + rescue_bonus_cap + fallback_bonus_cap)
    return accumulated
