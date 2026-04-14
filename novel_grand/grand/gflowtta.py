from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novel_grand.grand.baselines import llr_risk, run_baseline
from novel_grand.grand.search import Primitive, search_exact_syndrome
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import snapshot_feature_vector
from novel_grand.ldpc.pv_features import base_shortlist_scores, candidate_feature_matrix, state_feature_vector
from novel_grand.ldpc.tanner import TannerGraph, mask_to_indices, mask_to_numpy
from novel_grand.models.training import ActionPriorModel, SnapshotSelectorModel, StateValueModel


@dataclass(order=True)
class _BeamState:
    priority: float
    score: float
    snap_idx: int
    pattern_mask: int
    residual_mask: int
    depth: int
    action_kinds: tuple
    action_sizes: tuple



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



def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)



def _dynamic_pair_actions(
    candidate_bits: Sequence[int],
    action_scores: np.ndarray,
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
            if b1 == b2:
                continue
            cm2 = graph_exact.col_syndrome_masks[b2]
            pair_mask = cm1 ^ cm2
            new_w = (residual_mask ^ pair_mask).bit_count()
            delta = resid_w - new_w
            shared = len(checks1.intersection(graph_exact.bit_to_checks[b2]))
            score = float(action_scores[i] + action_scores[j]) + 0.10 * float(delta) + 0.05 * float(shared)
            rows.append((b1, b2, score))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[: max_pairs]



def _candidate_state_local_indices(shortlist: Sequence[int], pattern_mask: int) -> np.ndarray:
    out = np.zeros(len(shortlist), dtype=np.int8)
    for i, b in enumerate(shortlist):
        if (int(pattern_mask) >> int(b)) & 1:
            out[i] = 1
    return out



def _local_repair(
    trace: TraceResult,
    snapshot_idx: int,
    pattern_mask: int,
    shortlist: Sequence[int],
    shortlist_scores: np.ndarray,
    graph_exact: TannerGraph,
    cfg,
    *,
    query_budget: int,
) -> Dict:
    snapshot = trace.snapshots[snapshot_idx]
    hard0 = snapshot.hard ^ mask_to_numpy(int(pattern_mask), graph_exact.n)
    residual_mask = int(snapshot.syndrome_mask)
    for b in mask_to_indices(int(pattern_mask)):
        residual_mask ^= graph_exact.col_syndrome_masks[int(b)]

    resid_before = int(residual_mask).bit_count()
    valid0 = graph_exact.syndrome_mask(hard0) == 0
    exact0 = bool(np.array_equal(hard0.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    if residual_mask == 0:
        return {
            "queries": 1,
            "frontier_peak": 0,
            "pattern_mask": int(pattern_mask),
            "corrected_bits": hard0.astype(np.uint8),
            "selected_primitive_kinds": ["ai_gflow_complete"],
            "selected_primitive_sizes": [int(pattern_mask.bit_count())],
            "valid_codeword": bool(valid0),
            "success_exact": bool(exact0),
            "residual_weight_before_repair": int(resid_before),
            "residual_weight_after_repair": 0,
        }

    gcfg = cfg["grand"]
    local_bits = int(gcfg.get("gflow_local_repair_bits", 20))
    local_expand = int(gcfg.get("gflow_local_repair_expand_width", 12))
    local_max = int(gcfg.get("gflow_local_repair_max_toggles", 3))

    sel = set(mask_to_indices(int(pattern_mask)))
    order = [i for i in np.argsort(np.asarray(shortlist_scores, dtype=np.float32))[::-1] if int(shortlist[i]) not in sel]
    order = order[: local_bits]
    primitives: List[Primitive] = []
    for idx in order:
        bit = int(shortlist[idx])
        primitives.append(
            Primitive(
                name=f"toggle_{bit}",
                kind="bit",
                bit_indices=[bit],
                bit_mask=(1 << bit),
                syn_mask=graph_exact.col_syndrome_masks[bit],
                score=float(shortlist_scores[idx]),
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
        overlap_penalty=float(gcfg.get("overlap_penalty", 0.35)),
    )
    valid = graph_exact.syndrome_mask(res.corrected_bits) == 0
    exact = bool(np.array_equal(res.corrected_bits.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    resid_after = int(graph_exact.syndrome_mask(res.corrected_bits)).bit_count()
    sizes = [int(pattern_mask.bit_count())]
    if res.selected_primitive_sizes:
        sizes.extend(list(map(int, res.selected_primitive_sizes)))
    return {
        "queries": int(1 + res.queries),
        "frontier_peak": int(res.frontier_peak),
        "pattern_mask": int(pattern_mask ^ res.pattern_mask),
        "corrected_bits": res.corrected_bits.astype(np.uint8),
        "selected_primitive_kinds": ["ai_gflow_complete", "ai_gflow_localrepair"] + list(res.selected_primitive_kinds),
        "selected_primitive_sizes": sizes,
        "valid_codeword": bool(valid),
        "success_exact": bool(exact),
        "residual_weight_before_repair": int(resid_before),
        "residual_weight_after_repair": int(resid_after),
    }



def _search_round_on_snapshot(
    trace: TraceResult,
    snap_idx: int,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    action_model: ActionPriorModel,
    value_model: StateValueModel,
    cfg,
    *,
    query_budget: int,
    consensus_bias: np.ndarray | None,
    rng: np.random.Generator,
) -> Tuple[Dict | None, List[Dict], int, int]:
    gcfg = cfg["grand"]
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    bits_per_symbol = int(cfg["nr"]["bits_per_symbol"])
    fft_size = int(cfg["nr"]["fft_size"])
    shortlist_topk_bits = int(gcfg.get("gflow_shortlist_topk_bits", 96))
    policy_pool_bits = int(gcfg.get("gflow_policy_pool_bits", 24))
    beam_width = int(gcfg.get("gflow_beam_width", 8))
    max_depth = int(gcfg.get("gflow_max_depth", 10))
    max_pairs = int(gcfg.get("gflow_max_pairs_per_state", 6))
    max_pair_seed_bits = int(gcfg.get("gflow_pair_seed_bits", 8))
    top_complete = int(gcfg.get("gflow_top_complete_candidates", 10))
    value_weight = float(gcfg.get("gflow_value_weight", 0.45))
    llr_bias_weight = float(gcfg.get("gflow_llr_bias_weight", 0.25))
    depth_penalty = float(gcfg.get("gflow_depth_penalty", 0.03))
    stochastic_expand = int(gcfg.get("gflow_stochastic_expand", 1))

    bit_x, base_scores = base_shortlist_scores(
        trace,
        snap_idx,
        graph_struct,
        max_iter=max_iter,
        bits_per_symbol=bits_per_symbol,
        fft_size=fft_size,
    )
    shortlist = np.argsort(base_scores)[::-1][: shortlist_topk_bits].astype(int).tolist()
    short_scores = np.asarray([base_scores[int(b)] for b in shortlist], dtype=np.float32)
    short_bias = _rank01(short_scores)
    if consensus_bias is None:
        consensus_bias = np.zeros(len(shortlist), dtype=np.float32)
    else:
        consensus_bias = np.asarray(consensus_bias, dtype=np.float32).reshape(-1)
        if consensus_bias.size != len(shortlist):
            consensus_bias = np.zeros(len(shortlist), dtype=np.float32)

    snapshot = trace.snapshots[snap_idx]
    beams: List[_BeamState] = [
        _BeamState(
            priority=0.0,
            score=0.0,
            snap_idx=int(snap_idx),
            pattern_mask=0,
            residual_mask=int(snapshot.syndrome_mask),
            depth=0,
            action_kinds=(),
            action_sizes=(),
        )
    ]
    seen = {(int(snap_idx), 0)}
    frontier_peak = 1
    complete_states: List[_BeamState] = []

    for _ in range(max_depth):
        next_beams: List[_BeamState] = []
        for state in beams:
            # Allow STOP on non-empty states.
            if state.depth > 0:
                feat = state_feature_vector(
                    trace,
                    state.snap_idx,
                    graph_exact,
                    max_iter,
                    graph_struct.max_vn_degree,
                    selected_mask=int(state.pattern_mask),
                    residual_mask=int(state.residual_mask),
                ).reshape(1, -1)
                stop_value = float(value_model.predict_prob(feat)[0])
                stop_score = float(state.score + value_weight * stop_value - depth_penalty)
                complete_states.append(
                    _BeamState(
                        priority=-stop_score,
                        score=stop_score,
                        snap_idx=state.snap_idx,
                        pattern_mask=int(state.pattern_mask),
                        residual_mask=int(state.residual_mask),
                        depth=state.depth,
                        action_kinds=state.action_kinds + ("stop",),
                        action_sizes=state.action_sizes + (0,),
                    )
                )

            cand_bits = [b for b in shortlist if ((int(state.pattern_mask) >> int(b)) & 1) == 0][:policy_pool_bits]
            if not cand_bits:
                continue

            x_cand = candidate_feature_matrix(
                trace,
                state.snap_idx,
                graph_exact,
                graph_struct,
                max_iter=max_iter,
                bits_per_symbol=bits_per_symbol,
                fft_size=fft_size,
                selected_mask=int(state.pattern_mask),
                residual_mask=int(state.residual_mask),
                candidate_bits=cand_bits,
            )
            probs = action_model.predict_prob(x_cand)
            cand_bias = np.asarray([short_bias[shortlist.index(int(b))] for b in cand_bits], dtype=np.float32)
            cons = np.asarray([consensus_bias[shortlist.index(int(b))] for b in cand_bits], dtype=np.float32)
            scores = _logit(probs) + llr_bias_weight * cand_bias + cons
            order = np.argsort(scores)[::-1]

            chosen_idx = list(order[:beam_width])
            sample_pool = order[: max(beam_width * 2, 8)]
            if stochastic_expand > 0 and len(sample_pool) > 0:
                logits = scores[sample_pool] - float(np.max(scores[sample_pool]))
                ww = np.exp(logits)
                ww = ww / max(float(ww.sum()), 1e-8)
                n_s = min(stochastic_expand, len(sample_pool))
                extra = rng.choice(sample_pool, size=n_s, replace=False, p=ww)
                for idx in np.asarray(extra, dtype=int).tolist():
                    if idx not in chosen_idx:
                        chosen_idx.append(int(idx))

            top_bits_for_pairs = [int(cand_bits[i]) for i in order[: max_pair_seed_bits]]
            top_scores_for_pairs = np.asarray([float(scores[i]) for i in order[: max_pair_seed_bits]], dtype=np.float32)
            pair_actions = _dynamic_pair_actions(
                top_bits_for_pairs,
                top_scores_for_pairs,
                int(state.residual_mask),
                graph_exact,
                max_pairs=max_pairs,
                max_seed_bits=max_pair_seed_bits,
            )

            candidates: List[Tuple[str, Tuple[int, ...], float]] = []
            for idx in chosen_idx:
                b = int(cand_bits[int(idx)])
                colmask = graph_exact.col_syndrome_masks[b]
                delta = int(state.residual_mask).bit_count() - (int(state.residual_mask) ^ colmask).bit_count()
                local_score = float(scores[int(idx)]) + 0.08 * float(delta)
                candidates.append(("bit", (b,), local_score))
            for b1, b2, pair_score in pair_actions:
                candidates.append(("pair", (int(b1), int(b2)), float(pair_score)))
            candidates.sort(key=lambda x: x[2], reverse=True)

            for kind, bits, local_score in candidates[: max(beam_width, max_pairs)]:
                new_mask = int(state.pattern_mask)
                syn = 0
                for b in bits:
                    new_mask ^= (1 << int(b))
                    syn ^= graph_exact.col_syndrome_masks[int(b)]
                key = (state.snap_idx, int(new_mask))
                if key in seen:
                    continue
                seen.add(key)
                new_res = int(state.residual_mask) ^ int(syn)
                vfeat = state_feature_vector(
                    trace,
                    state.snap_idx,
                    graph_exact,
                    max_iter,
                    graph_struct.max_vn_degree,
                    selected_mask=int(new_mask),
                    residual_mask=int(new_res),
                ).reshape(1, -1)
                vprob = float(value_model.predict_prob(vfeat)[0])
                new_score = float(state.score + local_score + value_weight * vprob - depth_penalty * len(bits))
                next_beams.append(
                    _BeamState(
                        priority=-new_score,
                        score=new_score,
                        snap_idx=state.snap_idx,
                        pattern_mask=int(new_mask),
                        residual_mask=int(new_res),
                        depth=int(state.depth + len(bits)),
                        action_kinds=state.action_kinds + (kind,),
                        action_sizes=state.action_sizes + (len(bits),),
                    )
                )

        # de-duplicate and keep top beams
        dedup = {}
        for st in sorted(next_beams):
            key = (st.snap_idx, st.pattern_mask)
            if key in dedup:
                continue
            dedup[key] = st
            if len(dedup) >= beam_width:
                break
        beams = list(dedup.values())
        if not beams:
            break
        frontier_peak = max(frontier_peak, len(beams))

    if not complete_states:
        complete_states = beams

    # Exact verifier on complete candidates. Query budget counts only exact GRAND checks.
    remaining_budget = int(query_budget)
    candidate_rows: List[Dict] = []
    seen_masks = set()
    best_success = None
    for st in sorted(complete_states)[: max(top_complete, 1)]:
        if remaining_budget <= 0:
            break
        if int(st.pattern_mask) in seen_masks:
            continue
        seen_masks.add(int(st.pattern_mask))
        repaired = _local_repair(
            trace,
            snap_idx,
            int(st.pattern_mask),
            shortlist,
            short_scores,
            graph_exact,
            cfg,
            query_budget=remaining_budget,
        )
        remaining_budget -= int(repaired["queries"])
        out = {
            "score": float(st.score),
            "snap_idx": int(snap_idx),
            "pattern_mask": int(repaired["pattern_mask"]),
            "success_exact": bool(repaired["success_exact"]),
            "valid_codeword": bool(repaired["valid_codeword"]),
            "corrected_bits": repaired["corrected_bits"],
            "queries": int(repaired["queries"]),
            "frontier_peak": int(max(frontier_peak, repaired["frontier_peak"])),
            "primitive_kinds": ["ai_gflow"] + list(repaired["selected_primitive_kinds"]),
            "primitive_sizes": list(repaired["selected_primitive_sizes"]),
            "residual_weight_before_repair": int(repaired["residual_weight_before_repair"]),
            "residual_weight_after_repair": int(repaired["residual_weight_after_repair"]),
            "pattern_weight": int(repaired["pattern_mask"].bit_count()),
        }
        candidate_rows.append(out)
        if out["valid_codeword"]:
            exact = bool(np.array_equal(out["corrected_bits"].astype(np.uint8), trace.true_codeword.astype(np.uint8)))
            best_success = {
                "decoder": "gflowtta_grand",
                "selected_snapshot": int(snap_idx + 1),
                "selected_syndrome_weight": int(snapshot.syndrome_weight),
                "queries": int(query_budget - remaining_budget),
                "query_budget": int(query_budget),
                "frontier_peak": int(out["frontier_peak"]),
                "pattern_weight": int(out["pattern_weight"]),
                "success_exact": bool(exact),
                "valid_codeword": bool(out["valid_codeword"]),
                "undetected_error": bool(out["valid_codeword"] and not exact),
                "primitive_kinds": "|".join(["ai_gflowtta", ",".join(out["primitive_kinds"])]),
                "primitive_sizes": ",".join(map(str, out["primitive_sizes"])),
            }
            break

    return best_success, candidate_rows, int(query_budget - remaining_budget), int(frontier_peak)



def _consensus_bias_from_candidates(candidate_rows: Sequence[Dict], shortlist: Sequence[int], *, topn: int, strength: float) -> np.ndarray:
    k = len(shortlist)
    if k <= 0:
        return np.zeros((0,), dtype=np.float32)
    rows = list(candidate_rows)
    if not rows:
        return np.zeros((k,), dtype=np.float32)
    rows = sorted(rows, key=lambda r: (int(r.get("residual_weight_after_repair", 999999)), -float(r.get("score", -1e9))))[: max(topn, 1)]
    counts = np.zeros((k,), dtype=np.float32)
    index = {int(b): i for i, b in enumerate(shortlist)}
    for row in rows:
        for b in mask_to_indices(int(row.get("pattern_mask", 0))):
            if int(b) in index:
                counts[index[int(b)]] += 1.0
    if counts.sum() <= 0:
        return np.zeros((k,), dtype=np.float32)
    freq = counts / counts.sum()
    centered = freq - float(freq.mean())
    return (float(strength) * centered).astype(np.float32)



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



def run_gflowtta_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    action_model: ActionPriorModel,
    value_model: StateValueModel,
    cfg,
) -> Dict:
    """GFlow-TTA GRAND (CPU-feasible lite version).

    Stage 1: strong final-LLR guard on the final snapshot.
    Stage 2: learned snapshot selector chooses 1-2 snapshots.
    Stage 3: GFlow-inspired diverse set generation with cheap test-time adaptation.
    Stage 4: conservative best-syndrome LLR fallback.
    """
    base_cfg = copy.deepcopy(cfg)
    gcfg = base_cfg["grand"]
    q_main = int(gcfg["query_cap"])
    q_ai = int(gcfg.get("rescue_bonus_cap", max(2000, q_main // 2)))
    q_fb = int(gcfg.get("fallback_bonus_cap", max(1000, q_main // 4)))
    total_budget = q_main + q_ai + q_fb

    guard = run_baseline("final_llr_grand", trace, graph_exact, base_cfg, query_cap=q_main)
    guard = dict(guard)
    guard["decoder"] = "gflowtta_grand"
    guard["query_budget"] = int(total_budget)
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    snapshot_indices, _ = _snapshot_candidates(
        trace=trace,
        graph_struct=graph_struct,
        snapshot_model=snapshot_model,
        max_iter=max_iter,
        n=graph_exact.n,
        m=graph_exact.m,
        max_candidates=int(gcfg.get("gflow_snapshot_candidates", 2)),
    )

    if not snapshot_indices:
        snapshot_indices = [len(trace.snapshots) - 1]

    rng = np.random.default_rng(int(cfg["system"]["seed"]) + int(gcfg.get("gflow_seed_offset", 31)) + int(trace.stop_iteration))
    primary_frac = float(gcfg.get("gflow_primary_budget_fraction", 0.80))
    primary_budget = max(1, min(q_ai, int(round(primary_frac * q_ai))))
    secondary_budget = max(0, q_ai - primary_budget)

    all_candidates: List[Dict] = []
    total_ai_queries = 0
    frontier_peak = 0
    best_ai = None

    # Round 1 on primary selected snapshot.
    primary_idx = int(snapshot_indices[0])
    best_round, cand_rows, q_used, fp = _search_round_on_snapshot(
        trace=trace,
        snap_idx=primary_idx,
        graph_exact=graph_exact,
        graph_struct=graph_struct,
        action_model=action_model,
        value_model=value_model,
        cfg=cfg,
        query_budget=int(primary_budget),
        consensus_bias=None,
        rng=rng,
    )
    total_ai_queries += int(q_used)
    frontier_peak = max(frontier_peak, int(fp))
    all_candidates.extend(cand_rows)
    if best_round is not None and best_round.get("valid_codeword", False):
        best_ai = best_round

    # Round 2: cheap test-time adaptation via consensus bias on the same snapshot.
    if best_ai is None and primary_budget - q_used > 0:
        shortlist_for_bias = np.argsort(base_shortlist_scores(
            trace,
            primary_idx,
            graph_struct,
            max_iter=max_iter,
            bits_per_symbol=int(cfg["nr"]["bits_per_symbol"]),
            fft_size=int(cfg["nr"]["fft_size"]),
        )[1])[::-1][: int(gcfg.get("gflow_shortlist_topk_bits", 96))].astype(int).tolist()
        bias = _consensus_bias_from_candidates(
            cand_rows,
            shortlist_for_bias,
            topn=int(gcfg.get("gflow_tta_topn", 4)),
            strength=float(gcfg.get("gflow_tta_strength", 1.0)),
        )
        best_round2, cand_rows2, q_used2, fp2 = _search_round_on_snapshot(
            trace=trace,
            snap_idx=primary_idx,
            graph_exact=graph_exact,
            graph_struct=graph_struct,
            action_model=action_model,
            value_model=value_model,
            cfg=cfg,
            query_budget=int(max(primary_budget - q_used, 0)),
            consensus_bias=bias,
            rng=rng,
        )
        total_ai_queries += int(q_used2)
        frontier_peak = max(frontier_peak, int(fp2))
        all_candidates.extend(cand_rows2)
        if best_round2 is not None and best_round2.get("valid_codeword", False):
            best_ai = best_round2

    # Optional small secondary snapshot attempt if still unresolved.
    if best_ai is None and len(snapshot_indices) > 1 and secondary_budget > 0:
        best_sec, cand_rows_s, q_used_s, fp_s = _search_round_on_snapshot(
            trace=trace,
            snap_idx=int(snapshot_indices[1]),
            graph_exact=graph_exact,
            graph_struct=graph_struct,
            action_model=action_model,
            value_model=value_model,
            cfg=cfg,
            query_budget=int(secondary_budget),
            consensus_bias=None,
            rng=rng,
        )
        total_ai_queries += int(q_used_s)
        frontier_peak = max(frontier_peak, int(fp_s))
        all_candidates.extend(cand_rows_s)
        if best_sec is not None and best_sec.get("valid_codeword", False):
            best_ai = best_sec

    if best_ai is not None:
        best_ai = dict(best_ai)
        best_ai["queries"] = int(total_ai_queries)
        best_ai["query_budget"] = int(q_ai)
        best_ai["frontier_peak"] = int(max(frontier_peak, best_ai.get("frontier_peak", 0)))
        acc = _merge_results(guard, best_ai, name="gflowtta_grand", total_budget=total_budget)
        return acc

    # AI failed: report the best attempted snapshot for transparency.
    chosen_snap = int(snapshot_indices[0])
    ai_fail = {
        "decoder": "gflowtta_grand",
        "selected_snapshot": int(chosen_snap + 1),
        "selected_syndrome_weight": int(trace.snapshots[chosen_snap].syndrome_weight),
        "queries": int(total_ai_queries),
        "query_budget": int(q_ai),
        "frontier_peak": int(frontier_peak),
        "pattern_weight": 0,
        "success_exact": False,
        "valid_codeword": False,
        "undetected_error": False,
        "primitive_kinds": "ai_gflowtta",
        "primitive_sizes": "",
    }
    acc = _merge_results(guard, ai_fail, name="gflowtta_grand", total_budget=total_budget)

    fb = run_baseline("best_syndrome_llr_grand", trace, graph_exact, base_cfg, query_cap=q_fb)
    fb = dict(fb)
    fb["decoder"] = "gflowtta_grand"
    fb["query_budget"] = int(q_fb)
    fb["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr", str(fb.get("primitive_kinds", ""))] if x])
    return _merge_results(acc, fb, name="gflowtta_grand", total_budget=total_budget)
