from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import scipy.sparse as sp


def _indices_to_mask(indices: Iterable[int]) -> int:
    mask = 0
    for idx in indices:
        mask ^= 1 << int(idx)
    return mask


def mask_to_indices(mask: int) -> List[int]:
    out: List[int] = []
    cur = mask
    while cur:
        lsb = cur & -cur
        idx = lsb.bit_length() - 1
        out.append(idx)
        cur ^= lsb
    return out


def mask_to_numpy(mask: int, n: int) -> np.ndarray:
    arr = np.zeros(n, dtype=np.uint8)
    cur = mask
    while cur:
        lsb = cur & -cur
        idx = lsb.bit_length() - 1
        arr[idx] = 1
        cur ^= lsb
    return arr


@dataclass
class ComponentGroup:
    name: str
    bit_indices: List[int]


class TannerGraph:
    def __init__(self, pcm: sp.spmatrix):
        self.csr = pcm.tocsr().astype(np.uint8)
        self.csc = self.csr.tocsc().astype(np.uint8)
        self.m, self.n = self.csr.shape

        self.check_to_bits: List[List[int]] = []
        for i in range(self.m):
            start, end = self.csr.indptr[i], self.csr.indptr[i + 1]
            self.check_to_bits.append(self.csr.indices[start:end].astype(int).tolist())

        self.bit_to_checks: List[List[int]] = []
        self.col_syndrome_masks: List[int] = []
        for j in range(self.n):
            start, end = self.csc.indptr[j], self.csc.indptr[j + 1]
            checks = self.csc.indices[start:end].astype(int).tolist()
            self.bit_to_checks.append(checks)
            self.col_syndrome_masks.append(_indices_to_mask(checks))

        self.vn_degree = np.array([len(v) for v in self.bit_to_checks], dtype=np.int16)
        self.cn_degree = np.array([len(v) for v in self.check_to_bits], dtype=np.int16)
        self.max_vn_degree = int(self.vn_degree.max(initial=1))
        self.max_cn_degree = int(self.cn_degree.max(initial=1))

    def syndrome_mask(self, hard_bits: np.ndarray) -> int:
        hard_bits = np.asarray(hard_bits).astype(np.uint8).reshape(-1)
        if hard_bits.size != self.n:
            raise ValueError(f"Expected length {self.n}, got {hard_bits.size}")
        mask = 0
        ones = np.flatnonzero(hard_bits)
        for idx in ones:
            mask ^= self.col_syndrome_masks[int(idx)]
        return mask

    @staticmethod
    def syndrome_weight(mask: int) -> int:
        return int(mask.bit_count())

    def unsatisfied_check_indices(self, syndrome_mask: int) -> List[int]:
        return mask_to_indices(syndrome_mask)

    def unsatisfied_check_counts(self, syndrome_mask: int) -> np.ndarray:
        unsat = set(self.unsatisfied_check_indices(syndrome_mask))
        counts = np.zeros(self.n, dtype=np.int16)
        for j, checks in enumerate(self.bit_to_checks):
            c = 0
            for ch in checks:
                if ch in unsat:
                    c += 1
            counts[j] = c
        return counts

    def top_unsatisfied_check_groups(self, syndrome_mask: int, max_groups: int = 8) -> List[ComponentGroup]:
        unsat_checks = self.unsatisfied_check_indices(syndrome_mask)
        if not unsat_checks:
            return []
        groups: List[ComponentGroup] = []
        for ch in unsat_checks[:max_groups]:
            groups.append(ComponentGroup(name=f"check_{ch}", bit_indices=list(self.check_to_bits[ch])))
        return groups

    def unsatisfied_components(self, syndrome_mask: int, max_groups: int = 8, max_bits_per_group: int = 256) -> List[ComponentGroup]:
        unsat_checks = set(self.unsatisfied_check_indices(syndrome_mask))
        if not unsat_checks:
            return []

        visited_checks = set()
        visited_bits = set()
        groups: List[ComponentGroup] = []

        for start_check in list(unsat_checks):
            if start_check in visited_checks:
                continue

            queue_checks = [start_check]
            component_checks = set()
            component_bits = set()

            while queue_checks:
                ch = queue_checks.pop()
                if ch in visited_checks:
                    continue
                visited_checks.add(ch)
                component_checks.add(ch)

                for bit in self.check_to_bits[ch]:
                    component_bits.add(bit)
                    if bit in visited_bits:
                        continue
                    visited_bits.add(bit)
                    for nxt_check in self.bit_to_checks[bit]:
                        if nxt_check in unsat_checks and nxt_check not in visited_checks:
                            queue_checks.append(nxt_check)

            bits = sorted(component_bits)
            if bits:
                if len(bits) > max_bits_per_group:
                    bits = bits[:max_bits_per_group]
                groups.append(ComponentGroup(
                    name=f"component_{len(groups)}",
                    bit_indices=bits,
                ))
            if len(groups) >= max_groups:
                break

        return groups

    def contiguous_symbol_groups(self, bits_per_symbol: int, top_bit_indices: Sequence[int], max_groups: int = 8) -> List[ComponentGroup]:
        if bits_per_symbol <= 0:
            return []
        seen = set()
        groups: List[ComponentGroup] = []
        for bit in top_bit_indices:
            sym = int(bit) // bits_per_symbol
            if sym in seen:
                continue
            seen.add(sym)
            start = sym * bits_per_symbol
            stop = min(start + bits_per_symbol, self.n)
            groups.append(ComponentGroup(name=f"symbol_{sym}", bit_indices=list(range(start, stop))))
            if len(groups) >= max_groups:
                break
        return groups

    def subcarrier_groups(
        self,
        bits_per_symbol: int,
        fft_size: int,
        top_bit_indices: Sequence[int],
        max_groups: int = 8,
    ) -> List[ComponentGroup]:
        if bits_per_symbol <= 0 or fft_size <= 0:
            return []
        seen = set()
        groups: List[ComponentGroup] = []
        total_symbols = self.n // bits_per_symbol
        num_ofdm_symbols = max(total_symbols // fft_size, 1)
        for bit in top_bit_indices:
            sym = int(bit) // bits_per_symbol
            sc = sym % fft_size
            if sc in seen:
                continue
            seen.add(sc)
            bit_indices: List[int] = []
            for ofdm_idx in range(num_ofdm_symbols):
                sym_idx = ofdm_idx * fft_size + sc
                bit_start = sym_idx * bits_per_symbol
                bit_stop = min(bit_start + bits_per_symbol, self.n)
                bit_indices.extend(range(bit_start, bit_stop))
            groups.append(ComponentGroup(name=f"subcarrier_{sc}", bit_indices=bit_indices))
            if len(groups) >= max_groups:
                break
        return groups
