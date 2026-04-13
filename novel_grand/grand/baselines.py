from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from novel_grand.grand.search import Primitive, SearchResult, search_exact_syndrome
from novel_grand.ldpc.bp_trace import TraceResult
from novel_grand.ldpc.features import oracle_best_snapshot
from novel_grand.ldpc.tanner import TannerGraph


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = x.mean()
    sd = x.std()
    if sd < 1e-6:
        sd = 1.0
    return (x - mu) / sd


def llr_risk(snapshot) -> np.ndarray:
    return -np.abs(snapshot.posterior).astype(np.float32)


def unsat_llr_risk(snapshot) -> np.ndarray:
    return _zscore(snapshot.unsat_deg.astype(np.float32)) + _zscore(-np.abs(snapshot.posterior).astype(np.float32))


def _individual_primitives(graph: TannerGraph, scores: np.ndarray, topk: int) -> List[Primitive]:
    idx = np.argsort(scores)[::-1][:topk]
    out: List[Primitive] = []
    for j in idx:
        j = int(j)
        out.append(
            Primitive(
                name=f"bit_{j}",
                kind="bit",
                bit_indices=[j],
                bit_mask=(1 << j),
                syn_mask=graph.col_syndrome_masks[j],
                score=float(scores[j]),
            )
        )
    return out


def _run_search(graph: TannerGraph, snapshot, scores: np.ndarray, cfg, *, query_cap: int | None = None) -> SearchResult:
    gcfg = cfg["grand"]
    prims = _individual_primitives(graph, scores, int(gcfg["topk_bits"]))
    return search_exact_syndrome(
        n=graph.n,
        hard_bits=snapshot.hard,
        syndrome_mask=snapshot.syndrome_mask,
        primitives=prims,
        query_cap=int(gcfg["query_cap"] if query_cap is None else query_cap),
        max_primitives_in_pattern=int(gcfg["max_primitives_in_pattern"]),
        expand_width=int(gcfg["search_expand_width"]),
        overlap_penalty=float(gcfg["overlap_penalty"]),
    )


def _format_result(trace: TraceResult, graph: TannerGraph, snapshot, snap_idx: int, res: SearchResult, cfg, name: str) -> Tuple[Dict, Dict]:
    valid = graph.syndrome_mask(res.corrected_bits) == 0
    exact = bool(np.array_equal(res.corrected_bits.astype(np.uint8), trace.true_codeword.astype(np.uint8)))
    out = {
        "decoder": name,
        "selected_snapshot": snap_idx + 1,
        "selected_syndrome_weight": int(snapshot.syndrome_weight),
        "queries": int(res.queries),
        "query_budget": int(cfg["grand"]["query_cap"]),
        "frontier_peak": int(res.frontier_peak),
        "pattern_weight": int(res.pattern_mask.bit_count()),
        "success_exact": exact,
        "valid_codeword": bool(valid),
        "undetected_error": bool(valid and not exact),
        "primitive_kinds": ",".join(res.selected_primitive_kinds),
        "primitive_sizes": ",".join(map(str, res.selected_primitive_sizes)),
    }
    artifacts = {
        "selected_snapshot_index": int(snap_idx),
        "corrected_bits": res.corrected_bits.astype(np.uint8),
        "pattern_mask_int": int(res.pattern_mask),
    }
    return out, artifacts


def _total_tags_budget(cfg) -> int:
    gcfg = cfg["grand"]
    q_main = int(gcfg["query_cap"])
    q_rescue = int(gcfg.get("rescue_bonus_cap", max(2000, q_main // 2)))
    q_fb = int(gcfg.get("fallback_bonus_cap", max(1000, q_main // 4)))
    return q_main + q_rescue + q_fb


def _baseline_core(name: str, trace: TraceResult, graph: TannerGraph, cfg, *, query_cap: int | None = None) -> Tuple[Dict, Dict]:
    effective_query_cap = query_cap
    base_name = name
    if name == "final_llr_grand_capmatched":
        base_name = "final_llr_grand"
        effective_query_cap = _total_tags_budget(cfg)
    elif name == "best_syndrome_llr_grand_capmatched":
        base_name = "best_syndrome_llr_grand"
        effective_query_cap = _total_tags_budget(cfg)

    if base_name == "final_llr_grand":
        snap_idx = len(trace.snapshots) - 1
        scores = llr_risk(trace.snapshots[snap_idx])
    elif base_name == "best_syndrome_llr_grand":
        snap_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))
        scores = llr_risk(trace.snapshots[snap_idx])
    elif base_name == "best_syndrome_unsat_grand":
        snap_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))
        scores = unsat_llr_risk(trace.snapshots[snap_idx])
    elif base_name == "oracle_best_llr":
        snap_idx = oracle_best_snapshot(trace)
        scores = llr_risk(trace.snapshots[snap_idx])
    else:
        raise ValueError(f"Unknown baseline {name}")

    snapshot = trace.snapshots[snap_idx]
    res = _run_search(graph, snapshot, scores, cfg, query_cap=effective_query_cap)
    out, artifacts = _format_result(trace, graph, snapshot, snap_idx, res, cfg, name)
    if effective_query_cap is not None:
        out["query_budget"] = int(effective_query_cap)
    return out, artifacts




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


def run_guard_plus_best_syndrome(trace: TraceResult, graph: TannerGraph, cfg) -> Dict:
    gcfg = cfg["grand"]
    q_main = int(gcfg["query_cap"])
    q_extra = int(gcfg.get("rescue_bonus_cap", max(2000, q_main // 2))) + int(gcfg.get("fallback_bonus_cap", max(1000, q_main // 4)))
    total_budget = q_main + q_extra

    guard = run_baseline("final_llr_grand", trace, graph, cfg, query_cap=q_main)
    guard = dict(guard)
    guard["decoder"] = "guard_plus_best_syndrome"
    guard["primitive_kinds"] = "|".join([x for x in ["guard_final_llr", str(guard.get("primitive_kinds", ""))] if x])
    guard["query_budget"] = int(total_budget)
    if guard.get("valid_codeword", False):
        return guard

    fb = run_baseline("best_syndrome_llr_grand", trace, graph, cfg, query_cap=q_extra)
    fb = dict(fb)
    fb["decoder"] = "guard_plus_best_syndrome"
    fb["primitive_kinds"] = "|".join([x for x in ["fallback_best_syndrome_llr_nonai", str(fb.get("primitive_kinds", ""))] if x])
    merged = _merge_results(guard, fb, name="guard_plus_best_syndrome", total_budget=total_budget)
    return merged


def run_baseline(name: str, trace: TraceResult, graph: TannerGraph, cfg, *, query_cap: int | None = None) -> Dict:
    if name == "guard_plus_best_syndrome":
        return run_guard_plus_best_syndrome(trace, graph, cfg)
    out, _ = _baseline_core(name, trace, graph, cfg, query_cap=query_cap)
    return out


def run_baseline_detailed(name: str, trace: TraceResult, graph: TannerGraph, cfg, *, query_cap: int | None = None) -> Dict:
    if name == "guard_plus_best_syndrome":
        # Detailed artifacts are not currently used for this non-AI baseline.
        return run_guard_plus_best_syndrome(trace, graph, cfg)
    out, artifacts = _baseline_core(name, trace, graph, cfg, query_cap=query_cap)
    detailed = dict(out)
    detailed.update(artifacts)
    return detailed


def run_teacher_best_snapshot_llr(trace: TraceResult, graph: TannerGraph, cfg, *, query_cap: int | None = None) -> Dict:
    """Training-time teacher: choose the best LLR-ordered snapshot under budget.

    This is aligned with the actual rescue objective, unlike the raw-Hamming
    oracle snapshot. Exact success is preferred. Among exact rescues, fewer
    queries and smaller pattern weight win.
    """
    best_out = None
    best_art = None
    best_key = None
    for snap_idx, snapshot in enumerate(trace.snapshots):
        res = _run_search(graph, snapshot, llr_risk(snapshot), cfg, query_cap=query_cap)
        out, art = _format_result(trace, graph, snapshot, snap_idx, res, cfg, "teacher_best_snapshot_llr")
        key = (
            0 if out["success_exact"] else 1,
            int(out["queries"]),
            int(out["pattern_weight"]),
            int(snapshot.syndrome_weight),
            int(snap_idx),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_out = out
            best_art = art
    detailed = dict(best_out)
    detailed.update(best_art)
    return detailed
