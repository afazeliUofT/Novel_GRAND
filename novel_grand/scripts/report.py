from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from novel_grand.config import load_config, run_root
from novel_grand.sim.aggregate import load_frame_rows, summarize_frames
from novel_grand.utils.io import dataframe_to_csv, ensure_dir


def _ordered_groupby(df: pd.DataFrame, col: str):
    order = ["ldpc_only"] + sorted(x for x in df[col].unique() if x != "ldpc_only")
    for key in order:
        g = df[df[col] == key].sort_values("ebn0_db")
        if not g.empty:
            yield key, g


def _plot_metric(
    df_summary: pd.DataFrame,
    y_col: str,
    ylabel: str,
    out_path: Path,
    *,
    title: str | None = None,
    exclude_ldpc: bool = False,
) -> None:
    plt.figure()
    for decoder, g in _ordered_groupby(df_summary, "decoder"):
        if exclude_ldpc and decoder == "ldpc_only":
            continue
        plt.plot(g["ebn0_db"], g[y_col], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_net_fer(df_summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure()
    for decoder, g in _ordered_groupby(df_summary, "decoder"):
        y = g["net_frame_error_rate"].clip(lower=1e-12)
        plt.semilogy(g["ebn0_db"], y, marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Net frame error rate")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_gain_vs_ldpc(df_summary: pd.DataFrame, out_path: Path) -> None:
    ldpc = df_summary[df_summary["decoder"] == "ldpc_only"][["ebn0_db", "net_exact_success_rate"]].rename(
        columns={"net_exact_success_rate": "ldpc_net_exact_success_rate"}
    )
    merged = df_summary.merge(ldpc, on="ebn0_db", how="left")
    merged["net_success_gain_over_ldpc"] = (
        merged["net_exact_success_rate"] - merged["ldpc_net_exact_success_rate"]
    )

    plt.figure()
    for decoder, g in _ordered_groupby(merged, "decoder"):
        if decoder == "ldpc_only":
            continue
        plt.plot(g["ebn0_db"], g["net_success_gain_over_ldpc"], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Net exact-success gain over legacy LDPC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_markdown(df_summary: pd.DataFrame, out_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# TAGS-GRAND-Lite report")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "For rescue decoders, **conditional** success means success only on frames that "
        "legacy LDPC had already failed. **Net** success folds those rescues back into the "
        "whole-frame success rate and is the correct metric for comparing against legacy LDPC."
    )
    lines.append("")

    best_net = (
        df_summary.sort_values(["net_exact_success_rate", "decoder"], ascending=[False, True])
        .groupby("ebn0_db", as_index=False)
        .first()
    )
    dataframe_to_csv(best_net, out_dir / "best_decoder_by_ebn0.csv")

    lines.append("## Best decoder by Eb/N0 (net exact success)")
    lines.append("")
    for _, row in best_net.iterrows():
        lines.append(
            f"- `Eb/N0={row['ebn0_db']:.2f} dB`: **{row['decoder']}** "
            f"with net exact success `{row['net_exact_success_rate']:.6f}` "
            f"and average queries per original frame `{row['avg_queries_per_original_frame']:.2f}`."
        )
    lines.append("")

    lines.append("## Best conditional rescue decoder")
    lines.append("")
    rescue_only = df_summary[df_summary["decoder"] != "ldpc_only"].copy()
    if rescue_only.empty:
        lines.append("- No rescue-decoder rows were found.")
    else:
        best_cond = (
            rescue_only.sort_values(
                ["conditional_exact_success_rate", "decoder"],
                ascending=[False, True],
            )
            .groupby("ebn0_db", as_index=False)
            .first()
        )
        dataframe_to_csv(best_cond, out_dir / "best_rescue_decoder_by_ebn0.csv")
        for _, row in best_cond.iterrows():
            lines.append(
                f"- `Eb/N0={row['ebn0_db']:.2f} dB`: **{row['decoder']}** "
                f"conditional rescue success `{row['conditional_exact_success_rate']:.6f}`, "
                f"queries on invoked frames `{row['avg_queries_on_invoked_frames']:.2f}`."
            )
    lines.append("")

    lines.append("## Generated files")
    lines.append("")
    for name in [
        "frame_rows_all.csv",
        "summary_eval.csv",
        "best_decoder_by_ebn0.csv",
        "best_rescue_decoder_by_ebn0.csv",
        "net_exact_success_rate_vs_ebn0.png",
        "net_frame_error_rate_vs_ebn0.png",
        "conditional_rescue_success_rate_vs_ebn0.png",
        "avg_queries_on_invoked_frames_vs_ebn0.png",
        "avg_queries_per_original_frame_vs_ebn0.png",
        "net_success_gain_over_ldpc_vs_ebn0.png",
        "avg_selected_snapshot_vs_ebn0.png",
        "avg_frontier_peak_vs_ebn0.png",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    report_dir = ensure_dir(out_root / "reports")

    frame_paths = sorted((out_root / "eval" / "shards").glob("frame_rows_*.jsonl"))
    if not frame_paths:
        raise FileNotFoundError(
            "No evaluation shard jsonl files found. Run evaluate first."
        )

    df = load_frame_rows(frame_paths)
    df_summary = summarize_frames(df, quantiles=tuple(cfg["report"]["query_quantiles"]))

    dataframe_to_csv(df, report_dir / "frame_rows_all.csv")
    dataframe_to_csv(df_summary, report_dir / "summary_eval.csv")

    _plot_metric(
        df_summary,
        "net_exact_success_rate",
        "Net exact success rate",
        report_dir / "net_exact_success_rate_vs_ebn0.png",
    )
    _plot_net_fer(df_summary, report_dir / "net_frame_error_rate_vs_ebn0.png")
    _plot_metric(
        df_summary,
        "conditional_exact_success_rate",
        "Conditional rescue success rate",
        report_dir / "conditional_rescue_success_rate_vs_ebn0.png",
        exclude_ldpc=True,
    )
    _plot_metric(
        df_summary,
        "avg_queries_on_invoked_frames",
        "Average queries on invoked frames",
        report_dir / "avg_queries_on_invoked_frames_vs_ebn0.png",
        exclude_ldpc=True,
    )
    _plot_metric(
        df_summary,
        "avg_queries_per_original_frame",
        "Average queries per original frame",
        report_dir / "avg_queries_per_original_frame_vs_ebn0.png",
    )
    _plot_gain_vs_ldpc(
        df_summary,
        report_dir / "net_success_gain_over_ldpc_vs_ebn0.png",
    )
    _plot_metric(
        df_summary,
        "avg_selected_snapshot",
        "Average selected snapshot",
        report_dir / "avg_selected_snapshot_vs_ebn0.png",
    )
    _plot_metric(
        df_summary,
        "avg_frontier_peak",
        "Average search frontier peak",
        report_dir / "avg_frontier_peak_vs_ebn0.png",
        exclude_ldpc=True,
    )

    _write_markdown(df_summary, report_dir)

    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
