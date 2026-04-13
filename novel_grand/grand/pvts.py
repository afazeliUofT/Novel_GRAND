from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novel_grand.grand.baselines import run_baseline
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.pv_features import base_shortlist_scores, candidate_feature_matrix, state_feature_vector
from novel_grand.ldpc.tanner import TannerGraph, mask_to_numpy
from novel_grand.models.training import ActionPriorModel, SnapshotSelectorModel, StateValueModel


@dataclass(order=True)
class _State:
    priority: float
    score: float
    snap_idx: int
    pattern_mask: int
    residual_mask: int
    depth: int
    last_action_kind: str
    action_kinds: tuple
    action_sizes: tuple



def _snapshot_candidates(
    trace: TraceResult,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    max_iter: int,
    n: int,
    m: int,
    max_candidates: int = 3,
) -> Tuple[List[int], np.ndarray]:
    from novel_grand.ldpc.features import snapshot_feature_vector

    snap_feats = np.stack(
        [snapshot_feature_vector(s, max_iter, graph_struct.max_vn_degree, n, m) for s in trace.snapshots],
        axis=0,
    )
    pred = snapshot_model.predict(snap_feats).reshape(-1)
    order = list(np.argsort(pred))
    synd = np.asarray([s.syndrome_weight for s in trace.snapshots], dtype=np.float32)
    min_synd_idx = int(np.argmin(synd))
    final_idx = len(trace.snapshots) - 1

    candidates: List[int] = []
    for idx in [order[0], order[1] if len(order) > 1 else order[0], min_synd_idx, final_idx]:
        idx = int(idx)
        if idx not in candidates:
            candidates.append(idx)
        if len(candidates) >= max_candidates:
            break
    return candidates[:max_candidates], pred



def _root_bonus(pred: np.ndarray, idx: int) -> float:
    ranks = np.argsort(np.argsort(pred))
    return -0.10 * float(ranks[int(idx)])



def _dynamic_pair_actions(
    candidate_bits: Sequence[int],
    single_probs: np.ndarray,
    residual_mask: int,
    graph_exact: TannerGraph,
    *,
    max_pairs: int,
    max_seed_bits: int,
) -> List[Tuple[int, int, float]]:
    top_bits = list(candidate_bits[: max_seed_bits])
    rows: List[Tuple[int, int, float]] = []
    resid_w = residual_mask.bit_count()
    for i in range(len(top_bits)):
        b1 = int(top_bits[i])
        cm1 = graph_exact.col_syndrome_masks[b1]
        checks1 = set(graph_exact.bit_to_checks[b1])
        for j in range(i + 1, len(top_bits)):
            b2 = int(top_bits[j])
            cm2 = graph_exact.col_syndrome_masks[b2]
            if b1 == b2:
                continue
            pair_mask = cm1 ^ cm2
            new_w = (residual_mask ^ pair_mask).bit_count()
            delta = resid_w - new_w
            shared = len(checks1.intersection(graph_exact.bit_to_checks[b2]))
            score = float(single_probs[i] + single_probs[j]) + 0.08 * float(delta) + 0.05 * float(shared)
            rows.append((b1, b2, score))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:max_pairs]



def _search_on_snapshots(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_indices: Sequence[int],
    snapshot_pred: np.ndarray,
    policy_model: ActionPriorModel,
    value_model: StateValueModel,
    cfg,
    *,
    query_cap: int,
) -> Dict:
    gcfg = cfg["grand"]
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    bits_per_symbol = int(cfg["nr"]["bits_per_symbol"])
    fft_size = int(cfg["nr"]["fft_size"])
    shortlist_topk_bits = int(gcfg.get("flow_shortlist_topk_bits", 96))
    policy_pool_bits = int(gcfg.get("flow_policy_pool_bits", 16))
    expand_width = int(gcfg.get("flow_expand_width", 6))
    max_depth = int(gcfg.get("flow_max_depth", max(8, int(gcfg.get("max_primitives_in_pattern", 10)))))
    max_pairs = int(gcfg.get("flow_max_pairs_per_state", 6))
    max_pair_seed_bits = int(gcfg.get("flow_pair_seed_bits", 8))
    value_weight = float(gcfg.get("flow_value_weight", 0.55))
    delta_weight = float(gcfg.get("flow_delta_weight", 0.20))
    depth_penalty = float(gcfg.get("flow_depth_penalty", 0.03))

    precomp: Dict[int, Dict] = {}
    heap: List[_State] = []
    seen = set()
    frontier_peak = 0

    for snap_idx in snapshot_indices:
        snap_idx = int(snap_idx)
        snapshot = trace.snapshots[snap_idx]
        bit_x, base_scores = base_shortlist_scores(
            trace,
            snap_idx,
            graph_struct,
            max_iter=max_iter,
            bits_per_symbol=bits_per_symbol,
            fft_size=fft_size,
        )
        shortlist = np.argsort(base_scores)[::-1][:shortlist_topk_bits].astype(int).tolist()
        precomp[snap_idx] = {
            "snapshot": snapshot,
            "bit_x": bit_x,
            "shortlist": shortlist,
            "base_scores": base_scores,
        }
        root_feat = state_feature_vector(
            trace,
            snap_idx,
            graph_exact,
            max_iter,
            graph_struct.max_vn_degree,
            selected_mask=0,
            residual_mask=int(snapshot.syndrome_mask),
        ).reshape(1, -1)
        root_value = float(value_model.predict_prob(root_feat)[0])
        root_score = _root_bonus(snapshot_pred, snap_idx) + value_weight * root_value
        st = _State(
            priority=-root_score,
            score=root_score,
            snap_idx=snap_idx,
            pattern_mask=0,
            residual_mask=int(snapshot.syndrome_mask),
            depth=0,
            last_action_kind="root",
            action_kinds=(),
            action_sizes=(),
        )
        key = (snap_idx, 0)
        if key not in seen:
            seen.add(key)
            heapq.heappush(heap, st)

    queries = 0
    best = None
    while heap and queries < int(query_cap):
        frontier_peak = max(frontier_peak, len(heap))
        state = heapq.heappop(heap)
        if state.depth > 0:
            queries += 1
            if state.residual_mask == 0:
                snapshot = precomp[state.snap_idx]["snapshot"]
                corrected = snapshot.hard ^ mask_to_numpy(state.pattern_mask, graph_exact.n)
                valid = graph_exact.syndrome_mask(corrected) == 0
                exact = bool(np.array_equal(corrected.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
                best = {
                    "decoder": "flowsearch_grand",
                    "selected_snapshot": int(state.snap_idx + 1),
                    "selected_syndrome_weight": int(snapshot.syndrome_weight),
                    "queries": int(queries),
                    "query_budget": int(query_cap),
                    "frontier_peak": int(frontier_peak),
                    "pattern_weight": int(state.pattern_mask.bit_count()),
                    "success_exact": bool(exact),
                    "valid_codeword": bool(valid),
                    "undetected_error": bool(valid and not exact),
                    "primitive_kinds": "ai_flowsearch|" + ",".join(state.action_kinds),
                    "primitive_sizes": ",".join(map(str, state.action_sizes)),
                }
                break

        if state.depth >= max_depth:
            continue

        ctx = precomp[state.snap_idx]
        shortlist = [b for b in ctx["shortlist"] if ((state.pattern_mask >> int(b)) & 1) == 0][:policy_pool_bits]
        if not shortlist:
            continue
        x_cand = candidate_feature_matrix(
            trace,
            state.snap_idx,
            graph_exact,
            graph_struct,
            max_iter=max_iter,
            bits_per_symbol=bits_per_symbol,
            fft_size=fft_size,
            selected_mask=state.pattern_mask,
            residual_mask=state.residual_mask,
            candidate_bits=shortlist,
        )
        probs = policy_model.predict_prob(x_cand)
        order = np.argsort(probs)[::-1]
        top_single_idx = order[:expand_width]
        single_bits = [int(shortlist[i]) for i in top_single_idx]
        single_probs = np.asarray([float(probs[i]) for i in top_single_idx], dtype=np.float32)

        candidates: List[Tuple[str, Tuple[int, ...], float]] = []
        for b, p in zip(single_bits, single_probs):
            colmask = graph_exact.col_syndrome_masks[b]
            delta = state.residual_mask.bit_count() - (state.residual_mask ^ colmask).bit_count()
            score = np.log(max(float(p), 1e-6)) + delta_weight * float(delta) - depth_penalty
            candidates.append(("bit", (b,), float(score)))

        pair_actions = _dynamic_pair_actions(single_bits, single_probs, state.residual_mask, graph_exact, max_pairs=max_pairs, max_seed_bits=max_pair_seed_bits)
        for b1, b2, pair_score in pair_actions:
            candidates.append(("pair", (int(b1), int(b2)), float(pair_score) - depth_penalty))

        candidates.sort(key=lambda x: x[2], reverse=True)
        for kind, bits, local_score in candidates[: max(expand_width, max_pairs)]:
            new_mask = int(state.pattern_mask)
            syn_mask = 0
            for b in bits:
                new_mask ^= (1 << int(b))
                syn_mask ^= graph_exact.col_syndrome_masks[int(b)]
            key = (state.snap_idx, new_mask)
            if key in seen:
                continue
            seen.add(key)
            new_res = int(state.residual_mask) ^ int(syn_mask)
            value_feat = state_feature_vector(
                trace,
                state.snap_idx,
                graph_exact,
                max_iter,
                graph_struct.max_vn_degree,
                selected_mask=new_mask,
                residual_mask=new_res,
            ).reshape(1, -1)
            value_prob = float(value_model.predict_prob(value_feat)[0])
            new_score = float(state.score + local_score + value_weight * value_prob)
            heapq.heappush(
                heap,
                _State(
                    priority=-new_score,
                    score=new_score,
                    snap_idx=state.snap_idx,
                    pattern_mask=new_mask,
                    residual_mask=new_res,
                    depth=state.depth + len(bits),
                    last_action_kind=kind,
                    action_kinds=state.action_kinds + (kind,),
                    action_sizes=state.action_sizes + (len(bits),),
                ),
            )

    if best is not None:
        return best

    # Failure: return the first snapshot root for reporting consistency.
    first_snap = int(snapshot_indices[0]) if snapshot_indices else len(trace.snapshots) - 1
    first_snapshot = trace.snapshots[first_snap]
    return {
        "decoder": "flowsearch_grand",
        "selected_snapshot": int(first_snap + 1),
        "selected_syndrome_weight": int(first_snapshot.syndrome_weight),
        "queries": int(queries),
        "query_budget": int(query_cap),
        "frontier_peak": int(frontier_peak),
        "pattern_weight": 0,
        "success_exact": False,
        "valid_codeword": False,
        "undetected_error": False,
        "primitive_kinds": "ai_flowsearch",
        "primitive_sizes": "",
    }



def _merge_results(primary: Dict, secondary: Dict, *, name: str, total_budget: int) -> Dict:
    out = dict(secondary)
    out["decoder"] = name
    out["queries"] = int(primary.get("queries", 0)) + int(secondary.get("queries", 0))
    out["query_budget"] = int(total_budget)
    out["frontier_peak"] = int(max(primary.get("frontier_peak", 0), secondary.get("frontier_peak", 0)))
    pk1 = str(primary.get("primitive_kinds", ""))
    pk2 = str(secondary.get("primitive_kinds", ""))
    out["primitive_kinds"] = "|".join([x for x in [pk1, pk2] if x])
    ps1 = str(primary.get("primitive_sizes", ""))
    ps2 = str(secondary.get("primitive_sizes", ""))
    out["primitive_sizes"] = "|".join([x for x in [ps1, ps2] if x])
    return out



def run_flowsearch_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    policy_model: ActionPriorModel,
    value_model: StateValueModel,
    cfg,
) -> Dict:
    """New AI rescue path: multi-snapshot policy-value tree search.

    Stage 1: strong final-LLR guard on the final snapshot.
    Stage 2: flowsearch over 2-3 selected snapshots under a fixed AI budget.
    Stage 3: conservative best-syndrome LLR fallback.
    """
    base_cfg = copy.deepcopy(cfg)
    gcfg = base_cfg["grand"]
    q_main = int(gcfg["query_cap"])
    q_ai = int(gcfg.get("rescue_bonus_cap", max(1500, q_main // 2)))
    q_fb = int(gcfg.get("fallback_bonus_cap", max(1000, q_main // 4)))
    total_budget = q_main + q_ai + q_fb

    guard = run_baseline("final_llr_grand", trace, graph_exact, base_cfg, query_cap=q_main)
    guard = dict(guard)
    guard["decoder"] = "flowsearch_grand"
    guard["query_budget"] = int(total_budget)
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    snapshot_indices, snapshot_pred = _snapshot_candidates(
        trace=trace,
        graph_struct=graph_struct,
        snapshot_model=snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
        max_candidates=int(gcfg.get("flow_snapshot_candidates", 3)),
    )

    ai = _search_on_snapshots(
        trace=trace,
        graph_exact=graph_exact,
        graph_struct=graph_struct,
        snapshot_indices=snapshot_indices,
        snapshot_pred=snapshot_pred,
        policy_model=policy_model,
        value_model=value_model,
        cfg=cfg,
        query_cap=q_ai,
    )
    acc = _merge_results(guard, ai, name="flowsearch_grand", total_budget=total_budget)
    if ai.get("valid_codeword", False):
        return acc

    fb = run_baseline("best_syndrome_llr_grand", trace, graph_exact, base_cfg, query_cap=q_fb)
    fb = dict(fb)
    fb["decoder"] = "flowsearch_grand"
    fb["query_budget"] = int(q_fb)
    fb["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr", str(fb.get("primitive_kinds", ""))] if x])
    return _merge_results(acc, fb, name="flowsearch_grand", total_budget=total_budget)
