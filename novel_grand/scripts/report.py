from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from novel_grand.config import load_config, run_root
from novel_grand.sim.aggregate import load_frame_rows, summarize_frames
from novel_grand.utils.io import dataframe_to_csv, ensure_dir


def _plot_success(df_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    for decoder, g in df_summary.groupby("decoder"):
        plt.plot(g["ebn0_db"], g["success_exact_rate"], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Exact rescue / decode success rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_exact_rate_vs_ebn0.png", dpi=180)
    plt.close()


def _plot_queries(df_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    for decoder, g in df_summary.groupby("decoder"):
        if decoder == "ldpc_only":
            continue
        plt.plot(g["ebn0_db"], g["avg_queries"], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Average GRAND queries")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "avg_queries_vs_ebn0.png", dpi=180)
    plt.close()


def _plot_snapshot(df_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    for decoder, g in df_summary.groupby("decoder"):
        plt.plot(g["ebn0_db"], g["avg_selected_snapshot"], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Average selected snapshot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "avg_snapshot_vs_ebn0.png", dpi=180)
    plt.close()


def _write_markdown(df_summary: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    lines.append("# TAGS-GRAND-Lite report")
    lines.append("")
    lines.append("## Key observations")
    lines.append("")
    for decoder in sorted(df_summary["decoder"].unique()):
        d = df_summary[df_summary["decoder"] == decoder]
        if d.empty:
            continue
        best = d.sort_values("success_exact_rate", ascending=False).iloc[0]
        lines.append(
            f"- **{decoder}** best exact success rate = `{best['success_exact_rate']:.4f}` "
            f"at `Eb/N0={best['ebn0_db']:.2f} dB`, average queries `{best['avg_queries']:.1f}`."
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `summary_eval.csv`")
    lines.append("- `success_exact_rate_vs_ebn0.png`")
    lines.append("- `avg_queries_vs_ebn0.png`")
    lines.append("- `avg_snapshot_vs_ebn0.png`")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    report_dir = ensure_dir(out_root / "reports")

    frame_paths = sorted((out_root / "eval" / "shards").glob("frame_rows_*.jsonl"))
    if not frame_paths:
        raise FileNotFoundError("No evaluation shard jsonl files found. Run evaluate first.")

    df = load_frame_rows(frame_paths)
    df_summary = summarize_frames(df, quantiles=tuple(cfg["report"]["query_quantiles"]))
    dataframe_to_csv(df, report_dir / "frame_rows_all.csv")
    dataframe_to_csv(df_summary, report_dir / "summary_eval.csv")

    _plot_success(df_summary, report_dir)
    _plot_queries(df_summary, report_dir)
    _plot_snapshot(df_summary, report_dir)
    _write_markdown(df_summary, report_dir)

    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
