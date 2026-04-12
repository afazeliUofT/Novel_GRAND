from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novel_grand.config import load_config, run_root
from novel_grand.sim.aggregate import load_frame_rows, summarize_frames


def _line_plot(df: pd.DataFrame, x: str, y: str, out_path: Path, ylabel: str, include_ldpc: bool = True) -> None:
    plt.figure()
    for decoder, g in df.groupby("decoder", sort=True):
        if not include_ldpc and decoder == "ldpc_only":
            continue
        gg = g.sort_values(x)
        plt.plot(gg[x], gg[y], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def _gain_over_ldpc(df_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ebn0_db, g in df_summary.groupby("ebn0_db", sort=True):
        ldpc_row = g[g["decoder"] == "ldpc_only"]
        if ldpc_row.empty:
            continue
        ldpc_net = float(ldpc_row.iloc[0]["net_exact_success_rate"])
        for _, row in g.iterrows():
            rows.append(
                {
                    "ebn0_db": float(ebn0_db),
                    "decoder": str(row["decoder"]),
                    "net_success_gain_over_ldpc": float(row["net_exact_success_rate"] - ldpc_net),
                }
            )
    return pd.DataFrame(rows)



def _query_cap_hit_rates(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame_df.empty:
        return pd.DataFrame()
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        qmax = float(g["queries"].max()) if len(g) else 0.0
        hit_rate = float((g["queries"] >= qmax).mean()) if qmax > 0 else 0.0
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "decoder": str(decoder),
                "query_cap_hit_rate": hit_rate,
            }
        )
    return pd.DataFrame(rows)



def _primitive_usage(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame_df.empty:
        return pd.DataFrame()
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        bit = 0
        group = 0
        guard = 0
        fallback = 0
        ai = 0
        n = len(g)
        for s in g["primitive_kinds"].fillna(""):
            text = str(s)
            bit += int("bit" in text)
            group += int("group" in text)
            guard += int("guard_" in text)
            fallback += int("fallback_" in text)
            ai += int("ai_rescue" in text)
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "decoder": str(decoder),
                "bit_primitive_row_rate": bit / max(n, 1),
                "group_primitive_row_rate": group / max(n, 1),
                "guard_row_rate": guard / max(n, 1),
                "fallback_row_rate": fallback / max(n, 1),
                "ai_row_rate": ai / max(n, 1),
            }
        )
    return pd.DataFrame(rows)



def _write_markdown(df_summary: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    lines.append("# TAGS-GRAND report")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("For rescue decoders, **conditional** success means success only on frames that legacy LDPC had already failed.")
    lines.append("**Net** success folds those rescues back into the whole-frame success rate and is the correct metric for comparing against legacy LDPC.")
    lines.append("")
    lines.append("## Best decoder by Eb/N0 (net exact success)")
    for ebn0_db, g in df_summary.groupby("ebn0_db", sort=True):
        best = g.sort_values(["net_exact_success_rate", "avg_queries_per_original_frame"], ascending=[False, True]).iloc[0]
        lines.append(
            f"- `Eb/N0={ebn0_db:.2f} dB`: **{best['decoder']}** with net exact success `{best['net_exact_success_rate']:.6f}` "
            f"and average queries per original frame `{best['avg_queries_per_original_frame']:.2f}`."
        )
    lines.append("")
    lines.append("## Files")
    for name in [
        "summary_eval.csv",
        "frame_rows_all.csv",
        "net_exact_success_rate_vs_ebn0.png",
        "net_frame_error_rate_vs_ebn0.png",
        "conditional_rescue_success_rate_vs_ebn0.png",
        "avg_queries_on_invoked_frames_vs_ebn0.png",
        "avg_queries_per_original_frame_vs_ebn0.png",
        "net_success_gain_over_ldpc_vs_ebn0.png",
        "avg_selected_snapshot_vs_ebn0.png",
        "avg_frontier_peak_vs_ebn0.png",
        "query_cap_hit_rate_vs_ebn0.png",
        "primitive_usage_summary.csv",
        "query_cap_hit_summary.csv",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    report_dir = out_root / "reports"
    eval_dir = out_root / "eval" / "shards"
    report_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(str(p) for p in eval_dir.glob("frame_rows_worker*_snr*.jsonl"))
    frame_df = load_frame_rows(frame_paths)
    if frame_df.empty:
        raise RuntimeError(f"No evaluation rows found under {eval_dir}")

    frame_df = frame_df.sort_values(["ebn0_db", "decoder", "worker_id", "frame_idx"]).reset_index(drop=True)
    frame_df.to_csv(report_dir / "frame_rows_all.csv", index=False)

    quantiles = tuple(float(q) for q in cfg.get("report", {}).get("query_quantiles", [0.5, 0.9, 0.99]))
    df_summary = summarize_frames(frame_df, quantiles=quantiles)
    df_summary = df_summary.sort_values(["decoder", "ebn0_db"]).reset_index(drop=True)
    df_summary.to_csv(report_dir / "summary_eval.csv", index=False)

    df_gain = _gain_over_ldpc(df_summary)
    df_gain.to_csv(report_dir / "net_success_gain_over_ldpc.csv", index=False)
    df_qhit = _query_cap_hit_rates(frame_df)
    df_qhit.to_csv(report_dir / "query_cap_hit_summary.csv", index=False)
    df_pusage = _primitive_usage(frame_df)
    df_pusage.to_csv(report_dir / "primitive_usage_summary.csv", index=False)

    _line_plot(df_summary, "ebn0_db", "net_exact_success_rate", report_dir / "net_exact_success_rate_vs_ebn0.png", "Net exact success rate")
    _line_plot(df_summary, "ebn0_db", "net_frame_error_rate", report_dir / "net_frame_error_rate_vs_ebn0.png", "Net frame error rate")
    _line_plot(df_summary, "ebn0_db", "conditional_exact_success_rate", report_dir / "conditional_rescue_success_rate_vs_ebn0.png", "Conditional rescue success rate")
    _line_plot(df_summary, "ebn0_db", "avg_queries_on_invoked_frames", report_dir / "avg_queries_on_invoked_frames_vs_ebn0.png", "Avg queries on invoked frames", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_queries_per_original_frame", report_dir / "avg_queries_per_original_frame_vs_ebn0.png", "Avg queries per original frame", include_ldpc=False)
    _line_plot(df_gain, "ebn0_db", "net_success_gain_over_ldpc", report_dir / "net_success_gain_over_ldpc_vs_ebn0.png", "Net success gain over LDPC", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_selected_snapshot", report_dir / "avg_selected_snapshot_vs_ebn0.png", "Avg selected snapshot")
    _line_plot(df_summary, "ebn0_db", "avg_frontier_peak", report_dir / "avg_frontier_peak_vs_ebn0.png", "Avg frontier peak", include_ldpc=False)
    if not df_qhit.empty:
        _line_plot(df_qhit, "ebn0_db", "query_cap_hit_rate", report_dir / "query_cap_hit_rate_vs_ebn0.png", "Query-cap hit rate", include_ldpc=False)

    _write_markdown(df_summary, report_dir)


if __name__ == "__main__":
    main()
