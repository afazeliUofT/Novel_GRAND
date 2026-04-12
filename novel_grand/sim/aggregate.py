from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from novel_grand.utils.io import read_jsonl


def load_frame_rows(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in paths:
        for row in read_jsonl(path):
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_frames(
    df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> pd.DataFrame:
    """Summarize evaluation rows with both conditional and net metrics.

    Important distinction:
    - ``ldpc_only`` rows exist for *every* frame.
    - rescue decoder rows exist only for *legacy-detected-failure* frames.

    Therefore, rescue ``success_exact`` means *conditional rescue success*
    unless it is combined back with the legacy successes.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ebn0_db"] = df["ebn0_db"].astype(float)

    out_rows: List[Dict] = []

    for ebn0_db, g_snr in df.groupby("ebn0_db", sort=True):
        g_snr = g_snr.reset_index(drop=True)

        ldpc = g_snr[g_snr["decoder"] == "ldpc_only"].copy()
        n_original_frames = int(len(ldpc))
        if n_original_frames == 0:
            continue

        legacy_exact_successes = int(ldpc["success_exact"].sum())
        legacy_valid_codewords = int(ldpc["valid_codeword"].sum())
        legacy_undetected_errors = int(ldpc["undetected_error"].sum())
        legacy_detected_failures = int(ldpc["legacy_detected_failure"].sum())
        legacy_detected_failure_rate = legacy_detected_failures / n_original_frames

        for decoder, g in g_snr.groupby("decoder", sort=True):
            g = g.reset_index(drop=True)
            n_invoked_frames = int(len(g))
            queries = g["queries"].to_numpy(dtype=float)

            conditional_exact_successes = int(g["success_exact"].sum())
            conditional_valid_codewords = int(g["valid_codeword"].sum())
            conditional_undetected_errors = int(g["undetected_error"].sum())

            if decoder == "ldpc_only":
                net_exact_successes = legacy_exact_successes
                net_valid_codewords = legacy_valid_codewords
                net_undetected_errors = legacy_undetected_errors
            else:
                net_exact_successes = legacy_exact_successes + conditional_exact_successes
                net_valid_codewords = legacy_valid_codewords + conditional_valid_codewords
                net_undetected_errors = legacy_undetected_errors + conditional_undetected_errors

            row = {
                "ebn0_db": float(ebn0_db),
                "decoder": str(decoder),
                "n_original_frames": n_original_frames,
                "n_invoked_frames": n_invoked_frames,
                "legacy_detected_failures": legacy_detected_failures,
                "legacy_detected_failure_rate": float(legacy_detected_failure_rate),
                # Conditional / invoked-only metrics
                "conditional_exact_success_rate": float(conditional_exact_successes / max(n_invoked_frames, 1)),
                "conditional_valid_codeword_rate": float(conditional_valid_codewords / max(n_invoked_frames, 1)),
                "conditional_undetected_error_rate": float(conditional_undetected_errors / max(n_invoked_frames, 1)),
                "avg_queries_on_invoked_frames": float(np.mean(queries)) if queries.size else 0.0,
                "avg_pattern_weight": float(g["pattern_weight"].mean()) if n_invoked_frames else 0.0,
                "avg_selected_snapshot": float(g["selected_snapshot"].mean()) if n_invoked_frames else 0.0,
                "avg_frontier_peak": float(g["frontier_peak"].mean()) if n_invoked_frames else 0.0,
                # Net / whole-frame metrics
                "net_exact_success_rate": float(net_exact_successes / n_original_frames),
                "net_valid_codeword_rate": float(net_valid_codewords / n_original_frames),
                "net_undetected_error_rate": float(net_undetected_errors / n_original_frames),
                "net_frame_error_rate": float(1.0 - (net_exact_successes / n_original_frames)),
                "avg_queries_per_original_frame": float(queries.sum() / n_original_frames),
                "total_queries": float(queries.sum()),
                # Backward-compatible aliases
                "n_frames": n_invoked_frames,
                "success_exact_rate": float(conditional_exact_successes / max(n_invoked_frames, 1)),
                "valid_codeword_rate": float(conditional_valid_codewords / max(n_invoked_frames, 1)),
                "undetected_error_rate": float(conditional_undetected_errors / max(n_invoked_frames, 1)),
                "avg_queries": float(np.mean(queries)) if queries.size else 0.0,
            }

            for q in quantiles:
                if queries.size:
                    row[f"queries_q{int(100 * q)}"] = float(np.quantile(queries, q))
                else:
                    row[f"queries_q{int(100 * q)}"] = 0.0

            out_rows.append(row)

    return (
        pd.DataFrame(out_rows)
        .sort_values(["decoder", "ebn0_db"])
        .reset_index(drop=True)
    )
