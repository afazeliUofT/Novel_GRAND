from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from novel_grand.grand.baselines import _merge_results as merge_baseline_results
from novel_grand.grand.baselines import llr_risk, run_baseline, run_baseline_detailed
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import snapshot_feature_vector
from novel_grand.ldpc.tanner import TannerGraph, mask_to_numpy
from novel_grand.models.training import ActionPriorModel, SnapshotSelectorModel


@dataclass
class MemoryEntry:
    state_x: np.ndarray
    snapshot_idx: int
    risky_topk: List[int]
    correction_bits: List[int]
    teacher_queries: int
    ebn0_db: float



def load_memory_bank(path: str | Path) -> List[MemoryEntry]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[MemoryEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                MemoryEntry(
                    state_x=np.asarray(r["state_x"], dtype=np.float32),
                    snapshot_idx=int(r["snapshot_idx"]),
                    risky_topk=[int(x) for x in r.get("risky_topk", [])],
                    correction_bits=[int(x) for x in r.get("correction_bits", [])],
                    teacher_queries=int(r.get("teacher_queries", 0)),
                    ebn0_db=float(r.get("ebn0_db", 0.0)),
                )
            )
    return rows



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
    max_candidates: int = 4,
) -> List[int]:
    snap_feats = np.stack(
        [snapshot_feature_vector(s, max_iter, graph_struct.max_vn_degree, n, m) for s in trace.snapshots],
        axis=0,
    )
    pred = snapshot_model.predict(snap_feats).reshape(-1)
    n_snap = len(trace.snapshots)
    final_idx = n_snap - 1
    min_synd_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))
    order = list(np.argsort(pred))
    candidates: List[int] = []
    for idx in order[: max_candidates + 1]:
        if int(idx) not in candidates:
            candidates.append(int(idx))
    for idx in [min_synd_idx, final_idx]:
        if idx not in candidates:
            candidates.append(int(idx))
    return candidates[:max_candidates]



def _pair_features(query_state: np.ndarray, tmpl: MemoryEntry, query_risky: Sequence[int], n_snap: int) -> np.ndarray:
    q = np.asarray(query_state, dtype=np.float32).reshape(-1)
    t = np.asarray(tmpl.state_x, dtype=np.float32).reshape(-1)
    diff = np.abs(q - t)
    denom = max(float(np.linalg.norm(q) * np.linalg.norm(t)), 1e-6)
    cos = float(np.dot(q, t) / denom)
    qset = set(int(x) for x in query_risky)
    tset = set(int(x) for x in tmpl.risky_topk)
    overlap = float(len(qset & tset)) / max(len(qset | tset), 1)
    tmpl_size = float(len(tmpl.correction_bits))
    snap_gap = abs(int(tmpl.snapshot_idx)) / max(n_snap - 1, 1)
    return np.concatenate(
        [q, t, diff, np.array([cos, overlap, tmpl_size / 64.0, snap_gap], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)



def _current_risky_bits(snapshot, topk: int) -> List[int]:
    inv_abs_post = 1.0 / (np.abs(snapshot.posterior).astype(np.float32) + 1e-3)
    unsat = snapshot.unsat_deg.astype(np.float32)
    flips = snapshot.cumulative_flip_count.astype(np.float32)
    score = 0.65 * _rank01(inv_abs_post) + 0.25 * _rank01(unsat) + 0.10 * _rank01(flips)
    return np.argsort(score)[::-1][: int(topk)].astype(int).tolist()



def _retrieve_templates(
    query_state: np.ndarray,
    query_risky: Sequence[int],
    memory_bank: Sequence[MemoryEntry],
    template_ranker: ActionPriorModel | None,
    n_snap: int,
    cfg,
) -> List[tuple[float, MemoryEntry]]:
    gcfg = cfg["grand"]
    topk = int(gcfg.get("memo_topk_retrieve", 16))
    diversity_penalty = float(gcfg.get("memo_diversity_penalty", 0.10))
    rows: List[tuple[float, MemoryEntry, set[int]]] = []
    q = np.asarray(query_state, dtype=np.float32)
    qset = set(int(x) for x in query_risky)
    for tmpl in memory_bank:
        t = np.asarray(tmpl.state_x, dtype=np.float32)
        denom = max(float(np.linalg.norm(q) * np.linalg.norm(t)), 1e-6)
        cos = float(np.dot(q, t) / denom)
        tset = set(int(x) for x in tmpl.risky_topk)
        overlap = float(len(qset & tset)) / max(len(qset | tset), 1)
        snap_gap = abs(int(tmpl.snapshot_idx)) / max(n_snap - 1, 1)
        base = 0.60 * cos + 0.25 * overlap - 0.10 * snap_gap - 0.05 * min(len(tmpl.correction_bits), 64) / 64.0
        if template_ranker is not None:
            pf = _pair_features(q, tmpl, query_risky, n_snap)[None, :]
            base += 0.30 * float(template_ranker.predict_prob(pf)[0])
        rows.append((base, tmpl, set(tmpl.correction_bits)))
    rows.sort(key=lambda x: x[0], reverse=True)
    chosen: List[tuple[float, MemoryEntry]] = []
    chosen_sets: List[set[int]] = []
    for base, tmpl, cset in rows:
        penalty = 0.0
        for prev in chosen_sets:
            penalty = max(penalty, float(len(prev & cset)) / max(len(prev | cset), 1))
        score = base - diversity_penalty * penalty
        chosen.append((score, tmpl))
        chosen_sets.append(cset)
        if len(chosen) >= topk:
            break
    chosen.sort(key=lambda x: x[0], reverse=True)
    return chosen



def _mask_from_bits(bits: Sequence[int]) -> int:
    mask = 0
    for b in bits:
        mask ^= (1 << int(b))
    return int(mask)



def _try_mask(graph: TannerGraph, snapshot, trace: TraceResult, mask: int) -> tuple[bool, bool, np.ndarray]:
    residual = int(snapshot.syndrome_mask) ^ int(_mask_from_bits([]) if False else mask_to_syn(graph, mask))
    if residual != 0:
        corrected = snapshot.hard ^ mask_to_numpy(mask, graph.n)
        return False, False, corrected.astype(np.uint8)
    corrected = snapshot.hard ^ mask_to_numpy(mask, graph.n)
    exact = bool(np.array_equal(corrected.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    return True, exact, corrected.astype(np.uint8)



def mask_to_syn(graph: TannerGraph, mask: int) -> int:
    syn = 0
    cur = int(mask)
    while cur:
        lsb = cur & -cur
        idx = lsb.bit_length() - 1
        syn ^= int(graph.col_syndrome_masks[idx])
        cur ^= lsb
    return int(syn)



def _local_repair(
    graph: TannerGraph,
    snapshot,
    trace: TraceResult,
    base_mask: int,
    local_pool: Sequence[int],
    max_toggles: int,
    query_budget: int,
) -> tuple[bool, int, int, np.ndarray, str]:
    queries = 0
    best_corrected = snapshot.hard.copy().astype(np.uint8)
    local_bits = [int(b) for b in local_pool]
    n_local = len(local_bits)
    from itertools import combinations
    for k in range(1, max_toggles + 1):
        for combo in combinations(range(n_local), k):
            if queries >= query_budget:
                return False, queries, 0, best_corrected, ""
            toggle_bits = [local_bits[i] for i in combo]
            mask = int(base_mask) ^ _mask_from_bits(toggle_bits)
            queries += 1
            valid, exact, corrected = _try_mask(graph, snapshot, trace, mask)
            best_corrected = corrected
            if valid:
                return True, queries, mask, corrected, "ai_memory_localrepair"
    return False, queries, 0, best_corrected, ""



def _format_result(name: str, selected_snapshot: int, selected_syndrome_weight: int, queries: int, query_budget: int, frontier_peak: int, pattern_mask: int, valid: bool, exact: bool, primitive_kinds: Sequence[str], primitive_sizes: Sequence[int]) -> Dict:
    return {
        "decoder": name,
        "selected_snapshot": int(selected_snapshot),
        "selected_syndrome_weight": int(selected_syndrome_weight),
        "queries": int(queries),
        "query_budget": int(query_budget),
        "frontier_peak": int(frontier_peak),
        "pattern_weight": int(int(pattern_mask).bit_count()),
        "success_exact": bool(exact),
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": ",".join([str(x) for x in primitive_kinds if str(x)]),
        "primitive_sizes": ",".join(str(int(x)) for x in primitive_sizes),
    }



def run_memotta_grand(
    trace: TraceResult,
    graph_exact: TannerGraph,
    graph_struct: TannerGraph,
    snapshot_model: SnapshotSelectorModel,
    template_ranker: ActionPriorModel | None,
    memory_bank: Sequence[MemoryEntry],
    cfg,
) -> Dict:
    """Memory-Augmented Test-Time-Adapted GRAND.

    Stage 1: strong final-LLR guard.
    Stage 2: AI selects a snapshot.
    Stage 3: retrieve successful post-guard rescue templates from a persistent memory,
             apply one-step test-time adaptation to scores, generate diverse whole-set
             candidates, then exact syndrome verification plus local repair.
    Stage 4: conservative best-syndrome LLR fallback.
    """
    gcfg = cfg["grand"]
    q_guard = int(gcfg["query_cap"])
    q_ai = int(gcfg.get("rescue_bonus_cap", max(2000, q_guard // 2)))
    q_fb = int(gcfg.get("fallback_bonus_cap", max(1000, q_guard // 4)))
    q_total = q_guard + q_ai + q_fb

    guard = run_baseline("final_llr_grand", trace, graph_exact, cfg, query_cap=q_guard)
    guard = dict(guard)
    guard["decoder"] = "memotta_grand"
    guard["query_budget"] = q_total
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard.get("primitive_kinds", ""))] if x])
    if guard.get("valid_codeword", False):
        return guard

    n = graph_exact.n
    m = graph_exact.m
    max_iter = int(cfg["legacy_ldpc"]["num_iter"])
    snapshot_candidates = _snapshot_candidates(
        trace, graph_struct, snapshot_model, max_iter=max_iter, n=n, m=m,
        max_candidates=int(gcfg.get("memo_snapshot_candidates", 3)),
    )

    query_topk = int(gcfg.get("memo_risky_topk", 64))
    local_extra = int(gcfg.get("memo_local_pool_extra", 12))
    local_max_toggles = int(gcfg.get("memo_local_repair_max_toggles", 2))
    max_templates = int(gcfg.get("memo_templates_to_expand", 8))
    max_variants = int(gcfg.get("memo_variants_per_template", 3))
    tta_strength = float(gcfg.get("memo_tta_strength", 1.0))

    best_ai = None
    ai_queries = 0
    frontier_peak = 0

    for snap_idx in snapshot_candidates:
        if ai_queries >= q_ai:
            break
        snapshot = trace.snapshots[int(snap_idx)]
        state_x = snapshot_feature_vector(snapshot, max_iter, graph_struct.max_vn_degree, n, m)
        risky_bits = _current_risky_bits(snapshot, topk=query_topk)
        retrieved = _retrieve_templates(state_x, risky_bits, memory_bank, template_ranker, len(trace.snapshots), cfg)
        frontier_peak = max(frontier_peak, len(retrieved))

        for score, tmpl in retrieved[:max_templates]:
            if ai_queries >= q_ai:
                break
            tmpl_bits = [b for b in tmpl.correction_bits if 0 <= int(b) < n]
            if not tmpl_bits:
                continue
            # one-step test-time adaptation: bias toward current risky bits and same snapshot region
            adapted_bits = list(dict.fromkeys([b for b in tmpl_bits if b in risky_bits] + tmpl_bits))
            variants: List[List[int]] = []
            variants.append(adapted_bits)
            # diversify with current top risky bits not in template
            missing = [b for b in risky_bits if b not in adapted_bits][: 2 * max_variants]
            for j in range(max_variants - 1):
                bits = list(adapted_bits)
                if j < len(missing):
                    bits = list(dict.fromkeys(bits + [missing[j]]))
                if j + 1 < len(missing):
                    bits = [b for b in bits if b != tmpl_bits[min(j, len(tmpl_bits)-1)]] + [missing[j + 1]]
                variants.append(list(dict.fromkeys(bits)))

            template_bonus = tta_strength * max(0.0, score)
            for bits in variants:
                if ai_queries >= q_ai:
                    break
                mask = _mask_from_bits(bits)
                ai_queries += 1
                valid, exact, corrected = _try_mask(graph_exact, snapshot, trace, mask)
                if valid:
                    out = _format_result(
                        "memotta_grand",
                        selected_snapshot=int(snap_idx) + 1,
                        selected_syndrome_weight=int(snapshot.syndrome_weight),
                        queries=q_guard + ai_queries,
                        query_budget=q_total,
                        frontier_peak=frontier_peak,
                        pattern_mask=mask,
                        valid=True,
                        exact=exact,
                        primitive_kinds=["guard_final_llr", "ai_memory_template"],
                        primitive_sizes=[len(bits)],
                    )
                    return out
                # local repair around template plus current risky bits
                local_pool = list(dict.fromkeys(bits + risky_bits[:local_extra]))[: int(gcfg.get("memo_local_pool_cap", 20))]
                succ, used, mask2, corrected2, kind = _local_repair(
                    graph_exact,
                    snapshot,
                    trace,
                    mask,
                    local_pool,
                    max_toggles=local_max_toggles,
                    query_budget=max(q_ai - ai_queries, 0),
                )
                ai_queries += used
                if succ:
                    exact2 = bool(np.array_equal(corrected2.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
                    out = _format_result(
                        "memotta_grand",
                        selected_snapshot=int(snap_idx) + 1,
                        selected_syndrome_weight=int(snapshot.syndrome_weight),
                        queries=q_guard + ai_queries,
                        query_budget=q_total,
                        frontier_peak=frontier_peak,
                        pattern_mask=mask2,
                        valid=True,
                        exact=exact2,
                        primitive_kinds=["guard_final_llr", kind],
                        primitive_sizes=[int(mask2.bit_count())],
                    )
                    return out
        
    ai_fail = {
        "decoder": "memotta_grand",
        "selected_snapshot": int(snapshot_candidates[0]) + 1 if snapshot_candidates else len(trace.snapshots),
        "selected_syndrome_weight": int(trace.snapshots[int(snapshot_candidates[0])].syndrome_weight) if snapshot_candidates else int(trace.snapshots[-1].syndrome_weight),
        "queries": int(q_guard + ai_queries),
        "query_budget": int(q_total),
        "frontier_peak": int(frontier_peak),
        "pattern_weight": 0,
        "success_exact": False,
        "valid_codeword": False,
        "undetected_error": False,
        "primitive_kinds": "guard_final_llr|ai_memory",
        "primitive_sizes": "",
    }

    fb = run_baseline("best_syndrome_llr_grand", trace, graph_exact, cfg, query_cap=q_fb)
    fb = dict(fb)
    fb["decoder"] = "memotta_grand"
    fb["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr", str(fb.get("primitive_kinds", ""))] if x])
    fb["query_budget"] = q_total
    return merge_baseline_results(ai_fail, fb, name="memotta_grand", total_budget=q_total)
