from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np

from novel_grand.ldpc.tanner import mask_to_indices, mask_to_numpy


@dataclass
class Primitive:
    name: str
    kind: str
    bit_indices: List[int]
    bit_mask: int
    syn_mask: int
    score: float

    @property
    def size(self) -> int:
        return len(self.bit_indices)


@dataclass
class SearchResult:
    success: bool
    queries: int
    pattern_mask: int
    corrected_bits: np.ndarray
    selected_primitive_names: List[str]
    selected_primitive_kinds: List[str]
    selected_primitive_sizes: List[int]
    frontier_peak: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["corrected_bits"] = None
        d["pattern_weight"] = int(self.pattern_mask.bit_count())
        return d


@dataclass(order=True)
class _State:
    priority: float
    pattern_mask: int
    residual_mask: int
    score: float
    last_index: int
    primitive_indices: tuple


def search_exact_syndrome(
    n: int,
    hard_bits: np.ndarray,
    syndrome_mask: int,
    primitives: Sequence[Primitive],
    query_cap: int,
    max_primitives_in_pattern: int,
    expand_width: int,
    overlap_penalty: float,
) -> SearchResult:
    if syndrome_mask == 0:
        return SearchResult(
            success=True,
            queries=0,
            pattern_mask=0,
            corrected_bits=hard_bits.copy(),
            selected_primitive_names=[],
            selected_primitive_kinds=[],
            selected_primitive_sizes=[],
            frontier_peak=0,
        )

    heap: List[_State] = []
    seen = set()
    frontier_peak = 0

    for i, prim in enumerate(primitives):
        mask = prim.bit_mask
        state = _State(
            priority=-prim.score,
            pattern_mask=mask,
            residual_mask=syndrome_mask ^ prim.syn_mask,
            score=prim.score,
            last_index=i,
            primitive_indices=(i,),
        )
        if mask in seen:
            continue
        seen.add(mask)
        heapq.heappush(heap, state)

    queries = 0
    best_success = None

    while heap and queries < query_cap:
        frontier_peak = max(frontier_peak, len(heap))
        state = heapq.heappop(heap)
        queries += 1

        if state.residual_mask == 0:
            idxs = list(state.primitive_indices)
            selected = [primitives[i] for i in idxs]
            corrected = hard_bits ^ mask_to_numpy(state.pattern_mask, n)
            best_success = SearchResult(
                success=True,
                queries=queries,
                pattern_mask=state.pattern_mask,
                corrected_bits=corrected.astype(np.uint8),
                selected_primitive_names=[p.name for p in selected],
                selected_primitive_kinds=[p.kind for p in selected],
                selected_primitive_sizes=[p.size for p in selected],
                frontier_peak=frontier_peak,
            )
            break

        if len(state.primitive_indices) >= max_primitives_in_pattern:
            continue

        max_j = min(len(primitives), state.last_index + 1 + expand_width)
        for j in range(state.last_index + 1, max_j):
            prim = primitives[j]
            overlap = (state.pattern_mask & prim.bit_mask).bit_count()
            new_mask = state.pattern_mask ^ prim.bit_mask
            if new_mask in seen:
                continue
            seen.add(new_mask)
            new_score = state.score + prim.score - overlap_penalty * overlap / max(prim.size, 1)
            new_state = _State(
                priority=-new_score,
                pattern_mask=new_mask,
                residual_mask=state.residual_mask ^ prim.syn_mask,
                score=new_score,
                last_index=j,
                primitive_indices=state.primitive_indices + (j,),
            )
            heapq.heappush(heap, new_state)

    if best_success is not None:
        return best_success

    return SearchResult(
        success=False,
        queries=queries,
        pattern_mask=0,
        corrected_bits=hard_bits.copy(),
        selected_primitive_names=[],
        selected_primitive_kinds=[],
        selected_primitive_sizes=[],
        frontier_peak=frontier_peak,
    )
