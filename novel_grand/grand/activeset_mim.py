from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from novel_grand.grand.baselines import _merge_results as merge_baseline_results
from novel_grand.grand.baselines import llr_risk, run_baseline
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import bit_feature_matrix, snapshot_feature_vector
from novel_grand.ldpc.tanner import TannerGraph, mask_to_numpy
from novel_grand.models.training import BitRankerModel, SnapshotSelectorModel


@dataclass
class MiMResult:
    valid: bool
    exact: bool
    corrected_bits: np.ndarray
    pattern_mask: int
    pattern_weight: int
    queries: int
    solver_states: int
    selected_snapshot_index: int
    selected_syndrome_weight: int
    active_set_size: int
    confidence_gap: float


def mask_to_syn(graph: TannerGraph, mask: int) -> int:
    syn = 0
    cur = int(mask)
    while cur:
        lsb = cur & -cur
        idx = lsb.bit_length() - 1
        syn ^= int(graph.col_syndrome_masks[idx])
        cur ^= lsb
    return int(syn)


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(x.size, dtype=np.float32)
    return ranks / float(max(x.size - 1, 1))


def _snapshot_candidates(
    trace: TraceResult,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    cfg: Dict,
) -> List[int]:
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    n = graph_struct.n
    m = graph_struct.m
    snap_feats = np.stack(
        [
            snapshot_feature_vector(
                s,
                max_iter=max_iter,
                max_vn_degree=graph_struct.max_vn_degree,
                n=n,
                m=m,
            )
            for s in trace.snapshots
        ],
        axis=0,
    )
    pred = snapshot_model.predict(snap_feats).reshape(-1)
    order = list(np.argsort(pred))
    candidates: List[int] = []
    for idx in order[: int(cfg["grand"].get("astar_snapshot_candidates", 2))]:
        if int(idx) not in candidates:
            candidates.append(int(idx))
    final_idx = len(trace.snapshots) - 1
    min_synd_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))
    for idx in [min_synd_idx, final_idx]:
        if idx not in candidates:
            candidates.append(int(idx))
    return candidates[: int(cfg["grand"].get("astar_snapshot_candidates", 2))]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _score_bits(
    trace: TraceResult,
    snapshot_idx: int,
    graph: TannerGraph,
    bit_model: BitRankerModel,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, float]:
    nr_cfg = cfg["nr"]
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    x, _ = bit_feature_matrix(
        trace,
        snapshot_idx=snapshot_idx,
        max_iter=max_iter,
        max_vn_degree=graph.max_vn_degree,
        bits_per_symbol=int(nr_cfg["bits_per_symbol"]),
        fft_size=int(nr_cfg["fft_size"]),
    )
    logits = bit_model.predict_logits(x)
    snapshot = trace.snapshots[snapshot_idx]
    inv_abs_post = 1.0 / (np.abs(snapshot.posterior).astype(np.float32) + 1e-3)
    unsat = snapshot.unsat_deg.astype(np.float32)

    temps = cfg["grand"].get("astar_tta_temps", [0.8, 1.0, 1.25])
    probe_topk = int(cfg["grand"].get("astar_tta_probe_topk", 16))
    best_temp = 1.0
    best_obj = -1e18
    best_probs = None
    for temp in temps:
        probs = _sigmoid(logits / float(temp)).astype(np.float32)
        blend = 0.55 * _rank01(probs) + 0.25 * _rank01(unsat) + 0.20 * _rank01(inv_abs_post)
        top = np.argsort(blend)[::-1][:probe_topk]
        coverage = float(snapshot.unsat_deg[top].sum())
        confidence = float(probs[top].mean())
        compact = -0.05 * float(np.abs(snapshot.posterior[top]).mean())
        obj = coverage + confidence + compact
        if obj > best_obj:
            best_obj = obj
            best_temp = float(temp)
            best_probs = probs
    probs = np.asarray(best_probs, dtype=np.float32)
    blend = 0.55 * _rank01(probs) + 0.25 * _rank01(unsat) + 0.20 * _rank01(inv_abs_post)
    return blend.astype(np.float32), probs, best_temp


def _build_active_set(
    trace: TraceResult,
    snapshot_idx: int,
    graph: TannerGraph,
    scores: np.ndarray,
    cfg: Dict,
) -> List[int]:
    snapshot = trace.snapshots[snapshot_idx]
    nr_cfg = cfg["nr"]
    gcfg = cfg["grand"]
    base_topk = int(gcfg.get("astar_topk_bits", 14))
    active_cap = int(gcfg.get("astar_active_set_cap", 22))
    top_seed = np.argsort(scores)[::-1][:base_topk].astype(int).tolist()
    active: List[int] = []
    seen = set()

    def push(bit: int) -> None:
        if bit in seen or bit < 0 or bit >= graph.n:
            return
        seen.add(bit)
        active.append(int(bit))

    for b in top_seed:
        push(int(b))

    # Add bits from the most unsatisfied checks.
    for grp in graph.top_unsatisfied_check_groups(snapshot.syndrome_mask, max_groups=int(gcfg.get("astar_unsat_groups", 8))):
        for b in grp.bit_indices:
            push(int(b))
            if len(active) >= active_cap:
                break
        if len(active) >= active_cap:
            break

    # Add bits from the same modulation symbols and subcarriers as high-risk bits.
    if len(active) < active_cap:
        for grp in graph.contiguous_symbol_groups(int(nr_cfg["bits_per_symbol"]), top_seed, max_groups=8):
            for b in grp.bit_indices:
                push(int(b))
                if len(active) >= active_cap:
                    break
            if len(active) >= active_cap:
                break

    if len(active) < active_cap:
        for grp in graph.subcarrier_groups(int(nr_cfg["bits_per_symbol"]), int(nr_cfg["fft_size"]), top_seed, max_groups=4):
            for b in grp.bit_indices:
                push(int(b))
                if len(active) >= active_cap:
                    break
            if len(active) >= active_cap:
                break

    # Re-rank and trim.
    active = sorted(active, key=lambda b: float(scores[int(b)]), reverse=True)[:active_cap]
    return [int(b) for b in active]


def _enumerate_half(graph: TannerGraph, bits: Sequence[int], costs: Sequence[float], keep_per_syn: int) -> Tuple[Dict[int, List[Tuple[float, int]]], int]:
    n = len(bits)
    out: Dict[int, List[Tuple[float, int]]] = {}
    states = 0
    for mask_local in range(1 << n):
        syn = 0
        cost = 0.0
        pattern_mask = 0
        for i in range(n):
            if (mask_local >> i) & 1:
                bit = int(bits[i])
                syn ^= int(graph.col_syndrome_masks[bit])
                cost += float(costs[i])
                pattern_mask ^= (1 << bit)
        states += 1
        lst = out.setdefault(int(syn), [])
        lst.append((float(cost), int(pattern_mask)))
    for syn, lst in out.items():
        lst.sort(key=lambda x: x[0])
        out[syn] = lst[:keep_per_syn]
    return out, states


def _mim_candidates(
    graph: TannerGraph,
    syndrome_mask: int,
    active_bits: Sequence[int],
    bit_probs: np.ndarray,
    cfg: Dict,
) -> Tuple[List[Tuple[float, int]], int]:
    if not active_bits:
        return [], 0
    eps = 1e-4
    costs = [-float(np.log(np.clip(bit_probs[int(b)], eps, 1.0 - eps))) for b in active_bits]
    mid = len(active_bits) // 2
    left_bits = list(active_bits[:mid])
    right_bits = list(active_bits[mid:])
    left_costs = list(costs[:mid])
    right_costs = list(costs[mid:])
    keep_per_syn = int(cfg["grand"].get("astar_keep_per_syndrome", 8))
    left_map, states_l = _enumerate_half(graph, left_bits, left_costs, keep_per_syn)
    right_map, states_r = _enumerate_half(graph, right_bits, right_costs, keep_per_syn)
    candidates: List[Tuple[float, int]] = []
    for syn_l, left_list in left_map.items():
        target_r = int(syndrome_mask) ^ int(syn_l)
        right_list = right_map.get(target_r)
        if not right_list:
            continue
        for cl, ml in left_list:
            for cr, mr in right_list:
                candidates.append((float(cl + cr), int(ml ^ mr)))
    candidates.sort(key=lambda x: x[0])
    uniq: List[Tuple[float, int]] = []
    seen = set()
    for c, m in candidates:
        if m in seen:
            continue
        seen.add(m)
        uniq.append((c, m))
        if len(uniq) >= int(cfg["grand"].get("astar_top_candidates", 16)):
            break
    return uniq, int(states_l + states_r)


def _format_result(
    name: str,
    selected_snapshot: int,
    selected_syndrome_weight: int,
    queries: int,
    query_budget: int,
    solver_states: int,
    frontier_peak: int,
    pattern_mask: int,
    valid: bool,
    exact: bool,
    primitive_kinds: Sequence[str],
    primitive_sizes: Sequence[int],
) -> Dict:
    return {
        "decoder": name,
        "selected_snapshot": int(selected_snapshot),
        "selected_syndrome_weight": int(selected_syndrome_weight),
        "queries": int(queries),
        "query_budget": int(query_budget),
        "solver_states": int(solver_states),
        "frontier_peak": int(frontier_peak),
        "pattern_weight": int(int(pattern_mask).bit_count()),
        "success_exact": bool(exact),
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": ",".join([str(x) for x in primitive_kinds if str(x)]),
        "primitive_sizes": ",".join(str(int(x)) for x in primitive_sizes),
    }


def _try_snapshot(
    trace: TraceResult,
    graph: TannerGraph,
    snapshot_idx: int,
    bit_model: BitRankerModel,
    cfg: Dict,
    query_budget: int,
) -> MiMResult:
    snapshot = trace.snapshots[snapshot_idx]
    scores, probs, used_temp = _score_bits(trace, snapshot_idx, graph, bit_model, cfg)
    active_bits = _build_active_set(trace, snapshot_idx, graph, scores, cfg)
    candidates, solver_states = _mim_candidates(graph, snapshot.syndrome_mask, active_bits, probs, cfg)
    if not candidates:
        return MiMResult(
            valid=False,
            exact=False,
            corrected_bits=snapshot.hard.copy().astype(np.uint8),
            pattern_mask=0,
            pattern_weight=0,
            queries=0,
            solver_states=solver_states,
            selected_snapshot_index=int(snapshot_idx),
            selected_syndrome_weight=int(snapshot.syndrome_weight),
            active_set_size=len(active_bits),
            confidence_gap=0.0,
        )
    best_cost, best_mask = candidates[0]
    second_cost = candidates[1][0] if len(candidates) > 1 else (best_cost + 1.0)
    confidence_gap = float(second_cost - best_cost)
    accept_gap = float(cfg["grand"].get("astar_accept_gap", 0.15))
    corrected = snapshot.hard ^ mask_to_numpy(int(best_mask), graph.n)
    exact = bool(np.array_equal(corrected.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    valid = bool(graph.syndrome_mask(corrected) == 0)
    # Only accept a valid candidate if the margin is not too ambiguous.
    if valid and confidence_gap < accept_gap:
        valid = False
        exact = False
    return MiMResult(
        valid=valid,
        exact=exact,
        corrected_bits=corrected.astype(np.uint8),
        pattern_mask=int(best_mask),
        pattern_weight=int(int(best_mask).bit_count()),
        queries=1 if candidates else 0,
        solver_states=int(solver_states),
        selected_snapshot_index=int(snapshot_idx),
        selected_syndrome_weight=int(snapshot.syndrome_weight),
        active_set_size=len(active_bits),
        confidence_gap=confidence_gap,
    )


def run_activeset_mim_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    bit_model: BitRankerModel,
    cfg: Dict,
) -> Dict:
    gcfg = cfg["grand"]
    q_guard = int(gcfg.get("query_cap", 5000))
    q_fallback = int(gcfg.get("fallback_bonus_cap", 1500))
    total_budget = int(gcfg.get("total_capmatched_budget", q_guard + q_fallback))

    guard = run_baseline("final_llr_grand", trace, graph_exact, cfg, query_cap=q_guard)
    guard = dict(guard)
    guard["decoder"] = "activeset_mim_grand"
    guard["query_budget"] = int(total_budget)
    guard["solver_states"] = 0
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    candidates = _snapshot_candidates(trace, graph_struct, snapshot_model, cfg)
    best_ai: Dict | None = None
    best_key = None
    ai_total_queries = 0
    ai_total_states = 0
    used_candidates = 0
    for snap_idx in candidates:
        res = _try_snapshot(
            trace=trace,
            graph=graph_exact,
            snapshot_idx=int(snap_idx),
            bit_model=bit_model,
            cfg=cfg,
            query_budget=0,
        )
        ai_total_queries += int(res.queries)
        ai_total_states += int(res.solver_states)
        used_candidates += 1
        key = (
            0 if res.valid else 1,
            0 if res.exact else 1,
            -float(res.confidence_gap),
            int(res.pattern_weight),
            int(res.selected_snapshot_index),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_ai = {
                "decoder": "activeset_mim_grand",
                "selected_snapshot": int(res.selected_snapshot_index + 1),
                "selected_syndrome_weight": int(res.selected_syndrome_weight),
                "queries": int(ai_total_queries),
                "query_budget": int(total_budget),
                "solver_states": int(ai_total_states),
                "frontier_peak": int(res.active_set_size),
                "pattern_weight": int(res.pattern_weight),
                "success_exact": bool(res.exact),
                "valid_codeword": bool(res.valid),
                "undetected_error": bool(res.valid and not res.exact),
                "primitive_kinds": f"ai_activeset_mim|ai_selected_snapshot|ai_tta_temp|ai_candidates_{used_candidates}",
                "primitive_sizes": f"{int(res.active_set_size)},{int(res.pattern_weight)}",
            }
        if res.valid:
            break

    if best_ai is not None and bool(best_ai.get("valid_codeword", False)):
        return merge_baseline_results(guard, best_ai, name="activeset_mim_grand", total_budget=total_budget)

    fallback = run_baseline("best_syndrome_llr_grand", trace, graph_exact, cfg, query_cap=q_fallback)
    fallback = dict(fallback)
    fallback["decoder"] = "activeset_mim_grand"
    fallback["query_budget"] = int(total_budget)
    fallback["solver_states"] = int(ai_total_states)
    fallback["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr", str(fallback.get("primitive_kinds", ""))] if x])

    merged = merge_baseline_results(guard, fallback, name="activeset_mim_grand", total_budget=total_budget)
    merged["solver_states"] = int(ai_total_states)
    if best_ai is not None:
        merged["primitive_kinds"] = "|".join(
            [x for x in [str(guard.get("primitive_kinds", "")), str(best_ai.get("primitive_kinds", "")), str(fallback.get("primitive_kinds", ""))] if x]
        )
        merged["primitive_sizes"] = "|".join(
            [x for x in [str(guard.get("primitive_sizes", "")), str(best_ai.get("primitive_sizes", "")), str(fallback.get("primitive_sizes", ""))] if x]
        )
        if bool(best_ai.get("success_exact", False)):
            # If AI exact would have succeeded but did not pass confidence threshold, keep it as diagnostic only.
            merged["ai_shadow_success_exact"] = 1
    return merged
