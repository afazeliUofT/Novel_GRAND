from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from novel_grand.grand.baselines import run_baseline
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
    topk = max(int(gcfg.get("topk_bits", 96)), 128)
    max_groups = max(int(gcfg.get("max_group_count", 20)), 32)
    group_bonus_log_size = float(gcfg.get("group_bonus_log_size", 0.10))

    top_bits = np.argsort(scores)[::-1][:topk].astype(int).tolist()
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
                score=base_score + 0.08 * flip_bonus + 0.15 * unsat_bonus + size_bonus,
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
    final_idx = len(trace.snapshots) - 1
    model_order = list(np.argsort(pred))
    min_synd_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))

    candidates: List[int] = []

    def _add(idx: int) -> None:
        if 0 <= idx < len(trace.snapshots) and idx not in candidates:
            candidates.append(int(idx))

    _add(final_idx)
    _add(model_order[0])
    if len(model_order) > 1:
        _add(model_order[1])
    _add(min_synd_idx)

    chosen = final_idx
    for idx in candidates[1:]:
        if pred[idx] + 0.05 < pred[final_idx] and trace.snapshots[idx].syndrome_weight <= trace.snapshots[final_idx].syndrome_weight:
            chosen = idx
            break
        if trace.snapshots[idx].syndrome_weight + 8 < trace.snapshots[final_idx].syndrome_weight:
            chosen = idx
            break

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
    final_snapshot = trace.snapshots[-1]

    inv_abs_post = 1.0 / (np.abs(snapshot.posterior).astype(np.float32) + 1e-3)
    inv_abs_final = 1.0 / (np.abs(final_snapshot.posterior).astype(np.float32) + 1e-3)
    unsat = snapshot.unsat_deg.astype(np.float32)
    flips = snapshot.cumulative_flip_count.astype(np.float32)

    scores = (
        0.40 * _rank01(bit_prob)
        + 0.30 * _rank01(inv_abs_post)
        + 0.20 * _rank01(inv_abs_final)
        + 0.10 * _rank01(unsat + 0.25 * flips)
    ).astype(np.float32)
    return bit_prob, scores



def _search_stage(
    trace: TraceResult,
    snap_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    bit_model: BitRankerModel,
    cfg,
    query_cap: int,
) -> Dict:
    snapshot = trace.snapshots[snap_idx]
    _, bit_scores = _bit_scores_for_snapshot(trace, snap_idx, graph_struct, bit_model, cfg)
    topk_bits = max(int(cfg["grand"].get("topk_bits", 96)), 128)
    primitives: List[Primitive] = []
    top_bits = np.argsort(bit_scores)[::-1][:topk_bits]
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

    expand_width = max(int(cfg["grand"].get("search_expand_width", 10)), 24)
    max_prims = max(int(cfg["grand"].get("max_primitives_in_pattern", 10)), 12)
    res = search_exact_syndrome(
        n=graph_exact.n,
        hard_bits=snapshot.hard,
        syndrome_mask=snapshot.syndrome_mask,
        primitives=primitives,
        query_cap=int(query_cap),
        max_primitives_in_pattern=max_prims,
        expand_width=expand_width,
        overlap_penalty=float(cfg["grand"].get("overlap_penalty", 0.35)),
    )
    valid = graph_exact.syndrome_mask(res.corrected_bits) == 0
    exact = bool(np.array_equal(res.corrected_bits.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    return {
        "decoder": "tags_grand_lite",
        "selected_snapshot": snap_idx + 1,
        "selected_syndrome_weight": int(snapshot.syndrome_weight),
        "queries": int(res.queries),
        "query_budget": int(query_cap),
        "frontier_peak": int(res.frontier_peak),
        "pattern_weight": int(res.pattern_mask.bit_count()),
        "success_exact": bool(exact),
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": "ai_rescue" + ("|" + ",".join(res.selected_primitive_kinds) if res.selected_primitive_kinds else ""),
        "primitive_sizes": ",".join(map(str, res.selected_primitive_sizes)),
    }



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



def run_tags_grand_lite(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    bit_model: BitRankerModel,
    cfg,
) -> Dict:
    """Hybrid non-regression rescue.

    Stage 1: run the empirically strongest practical baseline (final-LLR GRAND).
    Stage 2: only if Stage 1 fails, spend an additional rescue budget on AI-selected
             snapshots using blended model + reliability scores and group primitives.
    Stage 3: if still unsuccessful, use a best-syndrome fallback budget.
    """
    base_cfg = copy.deepcopy(cfg)
    q_main = int(base_cfg["grand"]["query_cap"])
    rescue_bonus_cap = int(base_cfg["grand"].get("rescue_bonus_cap", q_main))
    fallback_bonus_cap = int(base_cfg["grand"].get("fallback_bonus_cap", max(1000, q_main // 2)))

    guard_base = run_baseline("final_llr_grand", trace, graph_exact, base_cfg)
    guard = dict(guard_base)
    guard["decoder"] = "tags_grand_lite"
    guard["query_budget"] = int(q_main + rescue_bonus_cap + fallback_bonus_cap)
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard_base.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    chosen_idx, candidates, _ = _snapshot_candidates(
        trace=trace,
        graph_struct=graph_struct,
        snapshot_model=snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
    )

    # Spend rescue budget across a small candidate snapshot portfolio.
    ai_caps = []
    n_ai = max(len(candidates), 1)
    base_share = max(512, rescue_bonus_cap // n_ai)
    remaining = rescue_bonus_cap
    for i in range(n_ai):
        if i == n_ai - 1:
            ai_caps.append(max(0, remaining))
        else:
            cap_i = min(base_share, remaining)
            ai_caps.append(cap_i)
            remaining -= cap_i

    accumulated = dict(guard)
    for snap_idx, stage_cap in zip(candidates, ai_caps):
        if stage_cap <= 0:
            continue
        ai_res = _search_stage(
            trace=trace,
            snap_idx=snap_idx,
            graph_exact=graph_exact,
            graph_struct=graph_struct,
            bit_model=bit_model,
            cfg=cfg,
            query_cap=stage_cap,
        )
        accumulated = _merge_results(accumulated, ai_res)
        if ai_res.get("valid_codeword", False):
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
