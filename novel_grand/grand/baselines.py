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


def _run_search(graph: TannerGraph, snapshot, scores: np.ndarray, cfg) -> SearchResult:
    gcfg = cfg["grand"]
    prims = _individual_primitives(graph, scores, int(gcfg["topk_bits"]))
    return search_exact_syndrome(
        n=graph.n,
        hard_bits=snapshot.hard,
        syndrome_mask=snapshot.syndrome_mask,
        primitives=prims,
        query_cap=int(gcfg["query_cap"]),
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


def _baseline_core(name: str, trace: TraceResult, graph: TannerGraph, cfg) -> Tuple[Dict, Dict]:
    if name == "final_llr_grand":
        snap_idx = len(trace.snapshots) - 1
        scores = llr_risk(trace.snapshots[snap_idx])
    elif name == "best_syndrome_llr_grand":
        snap_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))
        scores = llr_risk(trace.snapshots[snap_idx])
    elif name == "best_syndrome_unsat_grand":
        snap_idx = int(np.argmin([s.syndrome_weight for s in trace.snapshots]))
        scores = unsat_llr_risk(trace.snapshots[snap_idx])
    elif name == "oracle_best_llr":
        snap_idx = oracle_best_snapshot(trace)
        scores = llr_risk(trace.snapshots[snap_idx])
    else:
        raise ValueError(f"Unknown baseline {name}")

    snapshot = trace.snapshots[snap_idx]
    res = _run_search(graph, snapshot, scores, cfg)
    return _format_result(trace, graph, snapshot, snap_idx, res, cfg, name)


def run_baseline(name: str, trace: TraceResult, graph: TannerGraph, cfg) -> Dict:
    out, _ = _baseline_core(name, trace, graph, cfg)
    return out


def run_baseline_detailed(name: str, trace: TraceResult, graph: TannerGraph, cfg) -> Dict:
    out, artifacts = _baseline_core(name, trace, graph, cfg)
    detailed = dict(out)
    detailed.update(artifacts)
    return detailed


def run_teacher_best_snapshot_llr(trace: TraceResult, graph: TannerGraph, cfg) -> Dict:
    """Training-time teacher: choose the best LLR-ordered snapshot under budget.

    This is aligned with the actual rescue objective, unlike the raw-Hamming
    oracle snapshot. Exact success is preferred. Among exact rescues, fewer
    queries and smaller pattern weight win.
    """
    best_out = None
    best_art = None
    best_key = None
    for snap_idx, snapshot in enumerate(trace.snapshots):
        res = _run_search(graph, snapshot, llr_risk(snapshot), cfg)
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
