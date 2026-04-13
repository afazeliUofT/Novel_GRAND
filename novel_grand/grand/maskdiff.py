from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novel_grand.grand.baselines import run_baseline
from novel_grand.grand.search import Primitive, search_exact_syndrome
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.maskdiff_features import (
    MASK_VALUE,
    global_feature_vector,
    shortlist_with_required_bits,
    token_feature_matrix,
)
from novel_grand.ldpc.features import snapshot_feature_vector
from novel_grand.ldpc.pv_features import base_shortlist_scores
from novel_grand.ldpc.tanner import TannerGraph, mask_to_numpy
from novel_grand.models.maskdiff import MaskDiffModel
from novel_grand.models.training import SnapshotSelectorModel


@dataclass(order=True)
class _BeamState:
    priority: float
    score: float
    step_idx: int
    state_tuple: tuple



def _snapshot_candidates(
    trace: TraceResult,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    max_iter: int,
    n: int,
    m: int,
    max_candidates: int = 2,
) -> Tuple[List[int], np.ndarray]:
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



def _pattern_mask_from_shortlist(shortlist: np.ndarray, state: np.ndarray) -> int:
    mask = 0
    for local_idx, bit in enumerate(np.asarray(shortlist, dtype=np.int64).reshape(-1)):
        if int(state[local_idx]) > 0:
            mask ^= (1 << int(bit))
    return int(mask)



def _candidate_residual(snapshot_syndrome_mask: int, graph_exact: TannerGraph, shortlist: np.ndarray, state: np.ndarray) -> int:
    residual = int(snapshot_syndrome_mask)
    for local_idx, bit in enumerate(np.asarray(shortlist, dtype=np.int64).reshape(-1)):
        if int(state[local_idx]) > 0:
            residual ^= graph_exact.col_syndrome_masks[int(bit)]
    return int(residual)



def _state_logprob_increment(probs: np.ndarray, reveal_idx: np.ndarray, assigned: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float32)[reveal_idx], 1e-5, 1.0 - 1e-5)
    a = np.asarray(assigned, dtype=np.float32)
    return float(np.sum(a * np.log(p) + (1.0 - a) * np.log(1.0 - p)))



def _dedupe_top_beams(states: Sequence[_BeamState], width: int) -> List[_BeamState]:
    uniq = {}
    for st in sorted(states):
        if st.state_tuple in uniq:
            continue
        uniq[st.state_tuple] = st
        if len(uniq) >= width:
            break
    return list(uniq.values())



def _local_repair(
    trace: TraceResult,
    snapshot_idx: int,
    candidate_state: np.ndarray,
    candidate_probs: np.ndarray,
    shortlist: np.ndarray,
    shortlist_scores: np.ndarray,
    graph_exact: TannerGraph,
    cfg,
    *,
    query_budget: int,
) -> Dict:
    snapshot = trace.snapshots[snapshot_idx]
    pattern_mask = _pattern_mask_from_shortlist(shortlist, candidate_state)
    hard0 = snapshot.hard ^ mask_to_numpy(pattern_mask, graph_exact.n)
    residual_mask = _candidate_residual(int(snapshot.syndrome_mask), graph_exact, shortlist, candidate_state)

    valid0 = graph_exact.syndrome_mask(hard0) == 0
    exact0 = bool(np.array_equal(hard0.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    if residual_mask == 0:
        return {
            "queries": 1,
            "frontier_peak": 0,
            "pattern_mask": int(pattern_mask),
            "corrected_bits": hard0.astype(np.uint8),
            "selected_primitive_kinds": ["ai_maskdiff_sample"],
            "selected_primitive_sizes": [int(np.count_nonzero(candidate_state > 0))],
            "valid_codeword": bool(valid0),
            "success_exact": bool(exact0),
        }

    local_bits = int(cfg["grand"].get("mdiff_local_repair_bits", 20))
    local_expand = int(cfg["grand"].get("mdiff_local_repair_expand_width", 12))
    local_max = int(cfg["grand"].get("mdiff_local_repair_max_toggles", 3))

    entropy = 1.0 - np.abs(2.0 * np.asarray(candidate_probs, dtype=np.float32) - 1.0)
    combo = np.asarray(shortlist_scores, dtype=np.float32) + 0.25 * entropy
    order = np.argsort(combo)[::-1][:local_bits]
    primitives: List[Primitive] = []
    for j in order:
        bit = int(shortlist[int(j)])
        primitives.append(
            Primitive(
                name=f"toggle_{bit}",
                kind="bit",
                bit_indices=[bit],
                bit_mask=(1 << bit),
                syn_mask=graph_exact.col_syndrome_masks[bit],
                score=float(combo[int(j)]),
            )
        )

    res = search_exact_syndrome(
        n=graph_exact.n,
        hard_bits=hard0.astype(np.uint8),
        syndrome_mask=int(residual_mask),
        primitives=primitives,
        query_cap=int(max(query_budget - 1, 0)),
        max_primitives_in_pattern=int(local_max),
        expand_width=int(local_expand),
        overlap_penalty=float(cfg["grand"].get("overlap_penalty", 0.35)),
    )
    valid = graph_exact.syndrome_mask(res.corrected_bits) == 0
    exact = bool(np.array_equal(res.corrected_bits.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    kinds = ["ai_maskdiff_sample"]
    if res.selected_primitive_kinds:
        kinds.append("ai_maskdiff_localrepair")
        kinds.extend(list(res.selected_primitive_kinds))
    sizes = [int(np.count_nonzero(candidate_state > 0))]
    if res.selected_primitive_sizes:
        sizes.extend(list(map(int, res.selected_primitive_sizes)))
    return {
        "queries": int(1 + res.queries),
        "frontier_peak": int(res.frontier_peak),
        "pattern_mask": int(pattern_mask ^ res.pattern_mask),
        "corrected_bits": res.corrected_bits.astype(np.uint8),
        "selected_primitive_kinds": kinds,
        "selected_primitive_sizes": sizes,
        "valid_codeword": bool(valid),
        "success_exact": bool(exact),
    }



def _sample_candidates_for_snapshot(
    trace: TraceResult,
    snapshot_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    model: MaskDiffModel,
    cfg,
    *,
    consensus_bias: np.ndarray | None,
    budget: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict], int, int]:
    topk_bits = int(cfg["grand"].get("mdiff_shortlist_bits", 64))
    beam_width = int(cfg["grand"].get("mdiff_beam_width", 8))
    max_complete = int(cfg["grand"].get("mdiff_max_complete_candidates", 8))
    reveal_schedule = list(cfg["grand"].get("mdiff_reveal_schedule", [8, 8, 8, 8, 16, 16]))

    bit_x, shortlist, shortlist_scores = shortlist_with_required_bits(
        trace,
        snapshot_idx,
        graph_struct,
        cfg,
        topk_bits=topk_bits,
        required_bits=None,
    )
    base_short = bit_x[shortlist]
    k = int(shortlist.size)
    snapshot = trace.snapshots[snapshot_idx]

    beams: List[_BeamState] = [
        _BeamState(priority=0.0, score=0.0, step_idx=0, state_tuple=tuple([MASK_VALUE] * k))
    ]
    frontier_peak = 1

    for step_idx, reveal_count in enumerate(reveal_schedule, start=1):
        next_states: List[_BeamState] = []
        for beam in beams:
            state = np.asarray(beam.state_tuple, dtype=np.int8)
            masked_idx = np.flatnonzero(state < 0)
            if masked_idx.size == 0:
                next_states.append(beam)
                continue

            token_x = token_feature_matrix(base_short, state, step_frac=float(step_idx) / float(len(reveal_schedule)), consensus_bias=consensus_bias)
            global_x = global_feature_vector(
                trace,
                snapshot_idx,
                graph_exact,
                graph_struct,
                cfg,
                state=state,
                residual_mask=_candidate_residual(int(snapshot.syndrome_mask), graph_exact, shortlist, state),
                step_frac=float(step_idx) / float(len(reveal_schedule)),
            )
            probs = model.predict_prob(token_x[None, ...], global_x[None, ...])[0]
            conf = np.abs(probs[masked_idx] - 0.5)
            chosen = masked_idx[np.argsort(conf)[::-1][: min(int(reveal_count), masked_idx.size)]]

            det = state.copy()
            det_vals = (probs[chosen] >= 0.5).astype(np.int8)
            det[chosen] = det_vals
            det_score = beam.score + _state_logprob_increment(probs, chosen, det_vals)
            next_states.append(_BeamState(priority=-det_score, score=det_score, step_idx=step_idx, state_tuple=tuple(det.tolist())))

            sto = state.copy()
            clipped = np.clip(0.05 + 0.90 * probs[chosen], 0.02, 0.98)
            sto_vals = (rng.random(chosen.size) < clipped).astype(np.int8)
            sto[chosen] = sto_vals
            sto_score = beam.score + _state_logprob_increment(probs, chosen, sto_vals)
            next_states.append(_BeamState(priority=-sto_score, score=sto_score, step_idx=step_idx, state_tuple=tuple(sto.tolist())))

        beams = _dedupe_top_beams(next_states, width=beam_width)
        frontier_peak = max(frontier_peak, len(beams))

    complete_candidates: List[Dict] = []
    queries_used = 0
    for beam in beams:
        if queries_used >= budget or len(complete_candidates) >= max_complete:
            break
        state = np.asarray(beam.state_tuple, dtype=np.int8)
        if np.any(state < 0):
            token_x = token_feature_matrix(base_short, state, step_frac=1.0, consensus_bias=consensus_bias)
            global_x = global_feature_vector(
                trace,
                snapshot_idx,
                graph_exact,
                graph_struct,
                cfg,
                state=state,
                residual_mask=_candidate_residual(int(snapshot.syndrome_mask), graph_exact, shortlist, state),
                step_frac=1.0,
            )
            probs = model.predict_prob(token_x[None, ...], global_x[None, ...])[0]
            state[state < 0] = (probs[state < 0] >= 0.5).astype(np.int8)
        else:
            token_x = token_feature_matrix(base_short, state, step_frac=1.0, consensus_bias=consensus_bias)
            global_x = global_feature_vector(
                trace,
                snapshot_idx,
                graph_exact,
                graph_struct,
                cfg,
                state=state,
                residual_mask=_candidate_residual(int(snapshot.syndrome_mask), graph_exact, shortlist, state),
                step_frac=1.0,
            )
            probs = model.predict_prob(token_x[None, ...], global_x[None, ...])[0]

        repair_budget = min(int(cfg["grand"].get("mdiff_local_repair_budget_per_candidate", 32)), int(budget - queries_used))
        cand = _local_repair(
            trace,
            snapshot_idx,
            state,
            probs,
            shortlist,
            shortlist_scores[shortlist],
            graph_exact,
            cfg,
            query_budget=repair_budget,
        )
        cand["snapshot_idx"] = int(snapshot_idx)
        cand["score"] = float(beam.score)
        cand["state"] = state.astype(np.int8)
        queries_used += int(cand["queries"])
        complete_candidates.append(cand)

    complete_candidates.sort(
        key=lambda d: (
            0 if d["success_exact"] else 1,
            d["queries"],
            d["pattern_mask"].bit_count() if isinstance(d["pattern_mask"], int) else int(bin(int(d["pattern_mask"])).count("1")),
            -d["score"],
        )
    )
    return complete_candidates, int(queries_used), int(frontier_peak)



def _consensus_from_candidates(cands: Sequence[Dict], k: int, topn: int = 4) -> np.ndarray:
    if not cands:
        return np.zeros(k, dtype=np.float32)
    top = list(cands[: max(topn, 1)])
    arr = np.stack([np.asarray(c["state"], dtype=np.int8) for c in top], axis=0)
    # map {-1,0,1} -> {0,0,1}
    flips = (arr > 0).astype(np.float32)
    freq = flips.mean(axis=0)
    return (freq - 0.5).astype(np.float32)



def _best_ai_result(
    trace: TraceResult,
    graph_exact: TannerGraph,
    candidates: Sequence[Dict],
    *,
    queries: int,
    total_frontier_peak: int,
) -> Dict:
    if not candidates:
        snap_idx = len(trace.snapshots) - 1
        snapshot = trace.snapshots[snap_idx]
        return {
            "decoder": "maskdiff_grand",
            "selected_snapshot": int(snap_idx + 1),
            "selected_syndrome_weight": int(snapshot.syndrome_weight),
            "queries": int(queries),
            "query_budget": int(queries),
            "frontier_peak": int(total_frontier_peak),
            "pattern_weight": 0,
            "success_exact": False,
            "valid_codeword": False,
            "undetected_error": False,
            "primitive_kinds": "ai_maskdiff",
            "primitive_sizes": "",
        }

    best = candidates[0]
    snap_idx = int(best["snapshot_idx"])
    snapshot = trace.snapshots[snap_idx]
    corrected = np.asarray(best["corrected_bits"], dtype=np.uint8)
    valid = bool(graph_exact.syndrome_mask(corrected) == 0)
    exact = bool(np.array_equal(corrected, trace.true_codeword.astype(np.uint8)))
    return {
        "decoder": "maskdiff_grand",
        "selected_snapshot": int(snap_idx + 1),
        "selected_syndrome_weight": int(snapshot.syndrome_weight),
        "queries": int(queries),
        "query_budget": int(queries),
        "frontier_peak": int(total_frontier_peak),
        "pattern_weight": int(int(best["pattern_mask"]).bit_count()),
        "success_exact": bool(exact),
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": ",".join(list(best["selected_primitive_kinds"])),
        "primitive_sizes": ",".join(map(str, list(best["selected_primitive_sizes"]))),
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



def run_maskdiff_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    maskdiff_model: MaskDiffModel,
    cfg,
) -> Dict:
    """Verifier-guided masked diffusion rescue.

    Stage 1: strong deterministic final-LLR guard.
    Stage 2: masked-diffusion set generator on AI-selected snapshots with a
             lightweight consensus reconditioning round.
    Stage 3: conservative best-syndrome fallback.
    """
    base_cfg = copy.deepcopy(cfg)
    gcfg = base_cfg["grand"]
    q_main = int(gcfg["query_cap"])
    q_ai = int(gcfg.get("rescue_bonus_cap", max(2000, q_main // 2)))
    q_fb = int(gcfg.get("fallback_bonus_cap", max(1000, q_main // 4)))
    total_budget = q_main + q_ai + q_fb

    guard = run_baseline("final_llr_grand", trace, graph_exact, base_cfg, query_cap=q_main)
    guard = dict(guard)
    guard["decoder"] = "maskdiff_grand"
    guard["query_budget"] = int(total_budget)
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    seed_hint = int(np.round(float(np.abs(trace.llr_ch[: min(16, trace.llr_ch.size)]).sum()) * 1000.0))
    rng = np.random.default_rng(int(gcfg.get("mdiff_seed_offset", 17)) + int(seed_hint) + int(trace.stop_iteration))
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    snap_candidates, _ = _snapshot_candidates(
        trace,
        graph_struct,
        snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
        max_candidates=int(gcfg.get("mdiff_snapshot_candidates", 2)),
    )

    # Phase 1: multi-snapshot global proposals.
    all_ai_candidates: List[Dict] = []
    queries_ai = 0
    frontier_peak = 0
    primary_fraction = float(gcfg.get("mdiff_primary_snapshot_fraction", 0.70))
    per_budgets = []
    if snap_candidates:
        first_budget = int(round(primary_fraction * q_ai)) if len(snap_candidates) > 1 else int(q_ai)
        first_budget = max(1, min(first_budget, q_ai))
        remaining = max(q_ai - first_budget, 0)
        per_budgets.append(first_budget)
        if len(snap_candidates) > 1:
            share = max(1, remaining // (len(snap_candidates) - 1)) if remaining > 0 else 0
            for _ in snap_candidates[1:]:
                per_budgets.append(share)
    for snap_idx, budget in zip(snap_candidates, per_budgets):
        if queries_ai >= q_ai:
            break
        cands, q_used, fp = _sample_candidates_for_snapshot(
            trace,
            int(snap_idx),
            graph_exact,
            graph_struct,
            maskdiff_model,
            cfg,
            consensus_bias=None,
            budget=min(int(budget), int(q_ai - queries_ai)),
            rng=rng,
        )
        frontier_peak = max(frontier_peak, fp)
        queries_ai += int(q_used)
        all_ai_candidates.extend(cands)

    # Phase 2: consensus reconditioning on the strongest snapshot only.
    if snap_candidates and queries_ai < q_ai:
        bit_x, shortlist, _ = shortlist_with_required_bits(
            trace,
            int(snap_candidates[0]),
            graph_struct,
            cfg,
            topk_bits=int(cfg["grand"].get("mdiff_shortlist_bits", 64)),
            required_bits=None,
        )
        _ = bit_x  # shortlist length only
        consensus = _consensus_from_candidates(
            [c for c in all_ai_candidates if int(c["snapshot_idx"]) == int(snap_candidates[0])],
            k=int(shortlist.size),
            topn=int(gcfg.get("mdiff_consensus_topn", 4)),
        )
        if np.any(np.abs(consensus) > 1e-6):
            cands2, q_used2, fp2 = _sample_candidates_for_snapshot(
                trace,
                int(snap_candidates[0]),
                graph_exact,
                graph_struct,
                maskdiff_model,
                cfg,
                consensus_bias=float(gcfg.get("mdiff_consensus_strength", 1.0)) * consensus,
                budget=min(int(gcfg.get("mdiff_consensus_budget", max(256, q_ai // 5))), int(q_ai - queries_ai)),
                rng=rng,
            )
            frontier_peak = max(frontier_peak, fp2)
            queries_ai += int(q_used2)
            all_ai_candidates.extend(cands2)

    all_ai_candidates.sort(
        key=lambda d: (
            0 if d["success_exact"] else 1,
            d["queries"],
            int(d["pattern_mask"]).bit_count(),
            -float(d["score"]),
        )
    )
    ai = _best_ai_result(trace, graph_exact, all_ai_candidates, queries=queries_ai, total_frontier_peak=frontier_peak)
    ai["query_budget"] = int(q_ai)
    acc = _merge_results(guard, ai, name="maskdiff_grand", total_budget=total_budget)
    if ai.get("valid_codeword", False):
        return acc

    fb = run_baseline("best_syndrome_llr_grand", trace, graph_exact, base_cfg, query_cap=q_fb)
    fb = dict(fb)
    fb["decoder"] = "maskdiff_grand"
    fb["query_budget"] = int(q_fb)
    fb["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr", str(fb.get("primitive_kinds", ""))] if x])
    return _merge_results(acc, fb, name="maskdiff_grand", total_budget=total_budget)
