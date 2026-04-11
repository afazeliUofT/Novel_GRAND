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


def summarize_frames(df: pd.DataFrame, quantiles=(0.5, 0.9, 0.99)) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["ebn0_db", "decoder"]
    out_rows = []
    for (ebn0_db, decoder), g in df.groupby(group_cols):
        queries = g["queries"].to_numpy(dtype=float)
        row = {
            "ebn0_db": ebn0_db,
            "decoder": decoder,
            "n_frames": int(len(g)),
            "legacy_detected_failures": int(g["legacy_detected_failure"].sum()),
            "success_exact_rate": float(g["success_exact"].mean()),
            "valid_codeword_rate": float(g["valid_codeword"].mean()),
            "undetected_error_rate": float(g["undetected_error"].mean()),
            "avg_queries": float(np.mean(queries)),
            "avg_pattern_weight": float(g["pattern_weight"].mean()),
            "avg_selected_snapshot": float(g["selected_snapshot"].mean()),
            "avg_frontier_peak": float(g["frontier_peak"].mean()),
        }
        for q in quantiles:
            row[f"queries_q{int(100*q)}"] = float(np.quantile(queries, q))
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["decoder", "ebn0_db"]).reset_index(drop=True)
