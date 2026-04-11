from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from novel_grand.utils.io import ensure_dir


def _cache_dir(cfg: Dict) -> Path:
    repo_root = Path(cfg["repo_root"])
    return ensure_dir(repo_root / "outputs" / "_effective_code_cache")


def _cache_path(encoder, cfg: Dict) -> Path:
    bps = int(encoder.num_bits_per_symbol or 0)
    bg = getattr(encoder, "_bg", "na")  # pylint: disable=protected-access
    z = int(getattr(encoder, "z", 0))
    n_ldpc = int(getattr(encoder, "n_ldpc", encoder.n))
    name = f"ldpc_eff_pcm_k{int(encoder.k)}_n{int(encoder.n)}_nldpc{n_ldpc}_bps{bps}_bg{bg}_z{z}.npz"
    return _cache_dir(cfg) / name


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _lock_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".lock")


def _encode_basis_generator_matrix(encoder, chunk_size: int = 128) -> np.ndarray:
    k = int(encoder.k)
    rows: List[np.ndarray] = []
    eye = torch.eye(k, dtype=torch.float32, device=encoder.device)
    with torch.no_grad():
        for start in range(0, k, chunk_size):
            stop = min(k, start + chunk_size)
            u = eye[start:stop]
            c = encoder(u).detach().cpu().numpy().astype(np.uint8) & 1
            rows.append(c)
    g = np.concatenate(rows, axis=0)
    if g.shape != (k, int(encoder.n)):
        raise RuntimeError(f"Unexpected generator shape from encoder: {g.shape}, expected {(k, int(encoder.n))}.")
    return g


def _rref_binary(mat: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    a = (np.asarray(mat).astype(np.uint8) & 1).copy()
    m, n = a.shape
    row = 0
    pivot_cols: List[int] = []

    for col in range(n):
        if row >= m:
            break
        pivot_rel = np.flatnonzero(a[row:, col])
        if pivot_rel.size == 0:
            continue
        pivot = int(row + pivot_rel[0])
        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]

        other_rows = np.flatnonzero(a[:, col]).astype(int)
        other_rows = other_rows[other_rows != row]
        if other_rows.size > 0:
            a[other_rows] ^= a[row]

        pivot_cols.append(col)
        row += 1

    return a, pivot_cols


def _nullspace_basis_from_rref(rref: np.ndarray, pivot_cols: List[int]) -> np.ndarray:
    m, n = rref.shape
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]
    if len(pivot_cols) != m:
        raise RuntimeError(f"Generator matrix does not have full row rank: rank={len(pivot_cols)}, rows={m}.")
    h = np.zeros((len(free_cols), n), dtype=np.uint8)
    for idx, free_col in enumerate(free_cols):
        h[idx, free_col] = 1
        for r, pivot_col in enumerate(pivot_cols):
            h[idx, pivot_col] = rref[r, free_col] & 1
    return h


def _validate_pcm(g: np.ndarray, h: np.ndarray) -> None:
    prod = (g.astype(np.uint8) @ h.T.astype(np.uint8)) & 1
    if np.any(prod):
        raise RuntimeError("Computed effective parity-check matrix failed validation: G H^T != 0 over GF(2).")


def build_effective_pcm(encoder, chunk_size: int = 128) -> np.ndarray:
    g = _encode_basis_generator_matrix(encoder, chunk_size=chunk_size)
    rref, pivot_cols = _rref_binary(g)
    h = _nullspace_basis_from_rref(rref, pivot_cols)
    _validate_pcm(g, h)
    return h




def transmitted_mother_indices(encoder) -> np.ndarray:
    """Map transmitted bit positions back to mother-code column indices.

    This follows the LDPC5GEncoder call path for ``rv=None``:
    remove filler bits, skip the first ``2Z`` punctured positions,
    keep the first ``n`` bits, then apply the output interleaver.
    """
    n = int(encoder.n)
    k = int(encoder.k)
    k_ldpc = int(encoder.k_ldpc)
    z = int(encoder.z)

    if encoder.num_bits_per_symbol is not None:
        out_int = encoder.out_int.detach().cpu().numpy().astype(int)
        comp_idx = out_int.copy()
    else:
        comp_idx = np.arange(n, dtype=int)

    comp_idx = comp_idx + 2 * z
    mother_idx = comp_idx.copy()
    tail = comp_idx >= k
    mother_idx[tail] = comp_idx[tail] - k + k_ldpc
    return mother_idx.astype(int)


def projected_transmitted_pcm(encoder) -> sp.csr_matrix:
    """Sparse projected mother-code PCM restricted to transmitted positions.

    This is **not** an exact parity-check matrix for the rate-matched code.
    It is used only as a structure-preserving Tanner-graph proxy for features
    and group construction. Exact syndrome verification uses the cached
    effective PCM returned by :func:`load_or_build_effective_pcm`.
    """
    mother_idx = transmitted_mother_indices(encoder)
    pcm = encoder.pcm[:, mother_idx].tocsr().astype(np.uint8)
    nnz = np.diff(pcm.indptr)
    keep_rows = np.flatnonzero(nnz > 0)
    if keep_rows.size < pcm.shape[0]:
        pcm = pcm[keep_rows]
    return pcm


def _try_load_cached_pcm(cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        h = data["h"].astype(np.uint8)
    return h


def _write_cache_atomic(cache_path: Path, h: np.ndarray, meta: Dict) -> None:
    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, h=h.astype(np.uint8))
    os.replace(tmp_path, cache_path)
    _meta_path(cache_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_or_build_effective_pcm(encoder, cfg: Dict, chunk_size: int = 128, verbose: bool = False) -> sp.csr_matrix:
    cache_path = _cache_path(encoder, cfg)
    cached = _try_load_cached_pcm(cache_path)
    if cached is not None:
        return sp.csr_matrix(cached.astype(np.uint8))

    lock_path = _lock_path(cache_path)
    have_lock = False
    while not have_lock:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            have_lock = True
        except FileExistsError:
            cached = _try_load_cached_pcm(cache_path)
            if cached is not None:
                return sp.csr_matrix(cached.astype(np.uint8))
            time.sleep(0.25)

    try:
        cached = _try_load_cached_pcm(cache_path)
        if cached is not None:
            return sp.csr_matrix(cached.astype(np.uint8))

        if verbose:
            print(
                f"[effective-pcm] building cache for k={int(encoder.k)} n={int(encoder.n)} "
                f"n_ldpc={int(getattr(encoder, 'n_ldpc', encoder.n))} ...",
                flush=True,
            )
        h = build_effective_pcm(encoder, chunk_size=chunk_size)
        meta = {
            "k": int(encoder.k),
            "n": int(encoder.n),
            "n_ldpc": int(getattr(encoder, "n_ldpc", encoder.n)),
            "num_bits_per_symbol": int(encoder.num_bits_per_symbol or 0),
            "shape": [int(h.shape[0]), int(h.shape[1])],
            "cache_path": str(cache_path),
        }
        _write_cache_atomic(cache_path, h, meta)
        if verbose:
            print(f"[effective-pcm] wrote {cache_path}", flush=True)
        return sp.csr_matrix(h.astype(np.uint8))
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def prepare_effective_pcm_cache(cfg: Dict, encoder_factory, verbose: bool = False) -> Path:
    """Prepare the cached effective parity-check matrix.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    encoder_factory:
        Zero-argument callable returning an initialized Sionna encoder.
        Using a factory keeps this helper independent from the channel object.
    verbose:
        If ``True``, prints cache progress.
    """
    encoder = encoder_factory()
    cache_path = _cache_path(encoder, cfg)
    _ = load_or_build_effective_pcm(encoder, cfg, verbose=verbose)
    return cache_path
