from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novel_grand.config import load_config, run_root
from novel_grand.sim.aggregate import load_frame_rows, summarize_frames


BOOL_COLS = ["legacy_detected_failure", "success_exact", "valid_codeword", "undetected_error"]
PRIMARY_AI_DECODER = "activeset_mim_grand"


plt.rcParams.update(
    {
        "figure.figsize": (6.0, 4.0),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _normalize_boolish_series(s: pd.Series) -> pd.Series:
    mapping = {True: 1, False: 0, "True": 1, "False": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    return s.map(mapping).fillna(0).astype(int)


def _save_plot(fig, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _line_plot(df: pd.DataFrame, x: str, y: str, out_base: Path, ylabel: str, include_ldpc: bool = True, title: str | None = None) -> None:
    fig = plt.figure()
    for decoder, g in df.groupby("decoder", sort=True):
        if not include_ldpc and decoder == "ldpc_only":
            continue
        gg = g.sort_values(x)
        plt.plot(gg[x], gg[y], marker="o", label=decoder)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    _save_plot(fig, out_base)


def _line_plot_multi(df: pd.DataFrame, x: str, y_cols: list[str], out_base: Path, ylabel: str, title: str | None = None) -> None:
    fig = plt.figure()
    gg = df.sort_values(x)
    for col in y_cols:
        if col in gg.columns:
            plt.plot(gg[x], gg[col], marker="o", label=col)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    _save_plot(fig, out_base)


def _gain_over_reference(df_summary: pd.DataFrame, ref_decoder: str, out_col: str) -> pd.DataFrame:
    rows = []
    for ebn0_db, g in df_summary.groupby("ebn0_db", sort=True):
        ref_row = g[g["decoder"] == ref_decoder]
        if ref_row.empty:
            continue
        ref_net = float(ref_row.iloc[0]["net_exact_success_rate"])
        ref_q = float(ref_row.iloc[0].get("avg_queries_per_original_frame", 0.0))
        ref_s = float(ref_row.iloc[0].get("avg_solver_states_per_original_frame", 0.0))
        for _, row in g.iterrows():
            rows.append(
                {
                    "ebn0_db": float(ebn0_db),
                    "decoder": str(row["decoder"]),
                    out_col: float(row["net_exact_success_rate"] - ref_net),
                    "extra_queries_vs_reference": float(row.get("avg_queries_per_original_frame", 0.0) - ref_q),
                    "extra_solver_states_vs_reference": float(row.get("avg_solver_states_per_original_frame", 0.0) - ref_s),
                }
            )
    return pd.DataFrame(rows)


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


def _gap_to_oracle(df_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ebn0_db, g in df_summary.groupby("ebn0_db", sort=True):
        oracle = g[g["decoder"] == "oracle_best_llr"]
        if oracle.empty:
            continue
        oracle = oracle.iloc[0]
        for _, row in g.iterrows():
            rows.append(
                {
                    "ebn0_db": float(ebn0_db),
                    "decoder": str(row["decoder"]),
                    "net_success_gap_to_oracle": float(oracle["net_exact_success_rate"] - row["net_exact_success_rate"]),
                    "selected_snapshot_gap_to_oracle": float(row["avg_selected_snapshot"] - oracle["avg_selected_snapshot"]),
                }
            )
    return pd.DataFrame(rows)


def _query_efficiency(df_gain: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    merged = df_gain.merge(
        df_summary[["ebn0_db", "decoder", "avg_queries_per_original_frame", "avg_solver_states_per_original_frame"]],
        on=["ebn0_db", "decoder"],
        how="left",
    )
    q = merged["avg_queries_per_original_frame"].replace(0.0, np.nan)
    merged["net_success_gain_per_1000_queries"] = 1000.0 * merged["net_success_gain_over_ldpc"] / q
    merged["net_success_gain_per_1000_queries"] = merged["net_success_gain_per_1000_queries"].fillna(0.0)
    s = merged["avg_solver_states_per_original_frame"].replace(0.0, np.nan)
    merged["net_success_gain_per_1000_solver_states"] = 1000.0 * merged["net_success_gain_over_ldpc"] / s
    merged["net_success_gain_per_1000_solver_states"] = merged["net_success_gain_per_1000_solver_states"].fillna(0.0)
    return merged


def _query_cap_hit_rates(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame_df.empty:
        return pd.DataFrame()
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        budget = int(g["query_budget"].max()) if "query_budget" in g.columns else 0
        hit_rate = float((g["queries"] >= budget).mean()) if budget > 0 else 0.0
        rows.append({"ebn0_db": float(ebn0_db), "decoder": str(decoder), "query_budget": budget, "query_cap_hit_rate": hit_rate})
    return pd.DataFrame(rows)


def _primitive_usage(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        n = len(g)
        token_rows = [set([t.strip() for t in str(s).replace("|", ",").split(",") if t.strip()]) for s in g["primitive_kinds"].fillna("")]
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "decoder": str(decoder),
                "bit_primitive_row_rate": float(np.mean(["bit" in toks for toks in token_rows])) if n else 0.0,
                "group_primitive_row_rate": float(np.mean(["group" in toks for toks in token_rows])) if n else 0.0,
                "guard_row_rate": float(np.mean(["guard_final_llr" in toks for toks in token_rows])) if n else 0.0,
                "fallback_row_rate": float(np.mean([any(tok.startswith("fallback_") for tok in toks) for toks in token_rows])) if n else 0.0,
                "ai_row_rate": float(np.mean([any(tok.startswith("ai_") for tok in toks) for toks in token_rows])) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _ai_stage_contribution(frame_df: pd.DataFrame) -> pd.DataFrame:
    g_all = frame_df[frame_df["decoder"] == PRIMARY_AI_DECODER].copy()
    if g_all.empty:
        return pd.DataFrame()
    rows = []
    for ebn0_db, g in g_all.groupby("ebn0_db", sort=True):
        n = len(g)
        kinds = g["primitive_kinds"].fillna("").astype(str)
        success = g["success_exact"].astype(bool)
        guard_success = (success & kinds.str.contains("guard_final_llr") & ~kinds.str.contains("ai_") & ~kinds.str.contains("fallback_")).sum()
        ai_success = (success & kinds.str.contains("ai_") & ~kinds.str.contains("fallback_")).sum()
        fallback_success = (success & kinds.str.contains("fallback_")).sum()
        fail = n - (guard_success + ai_success + fallback_success)
        rows.append({
            "ebn0_db": float(ebn0_db),
            "guard_success_rate": guard_success / max(n, 1),
            "ai_success_rate": ai_success / max(n, 1),
            "fallback_success_rate": fallback_success / max(n, 1),
            "failure_rate": fail / max(n, 1),
        })
    return pd.DataFrame(rows)


def _ai_stage_visit(frame_df: pd.DataFrame) -> pd.DataFrame:
    g_all = frame_df[frame_df["decoder"] == PRIMARY_AI_DECODER].copy()
    if g_all.empty:
        return pd.DataFrame()
    rows = []
    for ebn0_db, g in g_all.groupby("ebn0_db", sort=True):
        kinds = g["primitive_kinds"].fillna("").astype(str)
        rows.append({
            "ebn0_db": float(ebn0_db),
            "guard_visit_rate": float(kinds.str.contains("guard_final_llr").mean()),
            "ai_visit_rate": float(kinds.str.contains("ai_").mean()),
            "fallback_visit_rate": float(kinds.str.contains("fallback_").mean()),
            "n_frames": int(len(g)),
        })
    return pd.DataFrame(rows)


def _worker_diversity(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()
    rows = []
    sig_cols = [
        "legacy_detected_failure", "selected_snapshot", "selected_syndrome_weight", "queries", "solver_states",
        "frontier_peak", "pattern_weight", "success_exact", "valid_codeword", "undetected_error",
    ]
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        unique_counts = []
        for _, gf in g.groupby("frame_idx", sort=True):
            sig = gf[sig_cols].astype(str).agg("|".join, axis=1)
            unique_counts.append(sig.nunique())
        unique_counts = np.asarray(unique_counts, dtype=float)
        rows.append({
            "ebn0_db": float(ebn0_db),
            "decoder": str(decoder),
            "avg_unique_outcomes_per_frame_slot": float(unique_counts.mean()) if unique_counts.size else 0.0,
            "perfect_duplication_rate": float(np.mean(unique_counts <= 1.0 + 1e-12)) if unique_counts.size else 0.0,
            "max_unique_outcomes_per_frame_slot": float(unique_counts.max()) if unique_counts.size else 0.0,
            "n_frame_slots": int(unique_counts.size),
        })
    return pd.DataFrame(rows)


def _publication_tables(df_summary: pd.DataFrame, df_gain_final_cap: pd.DataFrame, df_gain_guard_best: pd.DataFrame, out_dir: Path) -> None:
    keep = ["ldpc_only", "final_llr_grand", "final_llr_grand_capmatched", "guard_plus_best_syndrome", PRIMARY_AI_DECODER, "oracle_best_llr"]
    tbl = df_summary[df_summary["decoder"].isin(keep)].copy()
    cols = [
        "ebn0_db", "decoder", "net_exact_success_rate", "net_frame_error_rate", "conditional_exact_success_rate",
        "avg_queries_per_original_frame", "avg_solver_states_per_original_frame", "avg_selected_snapshot", "avg_frontier_peak",
    ]
    tbl[cols].to_csv(out_dir / "publication_main_table.csv", index=False)
    if not df_gain_final_cap.empty and not df_gain_guard_best.empty:
        delta = df_gain_final_cap[["ebn0_db", "decoder", "net_success_gain_over_final_capmatched", "extra_queries_vs_reference", "extra_solver_states_vs_reference"]].merge(
            df_gain_guard_best[["ebn0_db", "decoder", "net_success_gain_over_guard_plus_best_syndrome"]],
            on=["ebn0_db", "decoder"], how="left"
        )
        delta.to_csv(out_dir / "publication_delta_table.csv", index=False)


def _write_markdown(df_summary: pd.DataFrame, df_stage: pd.DataFrame, df_stage_visit: pd.DataFrame, df_div: pd.DataFrame, df_qhit: pd.DataFrame, out_dir: Path, df_gain_final_cap: pd.DataFrame, df_gain_guard_best: pd.DataFrame) -> None:
    lines = [f"# Active-Set MiM GRAND report", "", "## Interpretation", "For rescue decoders, **conditional** success is measured only on legacy-LDPC failures.", "**Net** success folds those rescues back into the whole-frame success rate and is the metric that should be compared against legacy LDPC.", "", "## Best decoder by Eb/N0 (net exact success)"]
    for ebn0_db, g in df_summary.groupby("ebn0_db", sort=True):
        best = g.sort_values(["net_exact_success_rate", "avg_queries_per_original_frame"], ascending=[False, True]).iloc[0]
        lines.append(f"- `Eb/N0={ebn0_db:.2f} dB`: **{best['decoder']}** with net exact success `{best['net_exact_success_rate']:.6f}` and average queries per original frame `{best['avg_queries_per_original_frame']:.2f}`.")
    if not df_stage.empty:
        lines += ["", "## AI stage contribution (conditional on LDPC-detected failures)"]
        for _, row in df_stage.sort_values("ebn0_db").iterrows():
            lines.append(f"- `Eb/N0={row['ebn0_db']:.2f} dB`: guard `{row['guard_success_rate']:.6f}`, ai `{row['ai_success_rate']:.6f}`, fallback `{row['fallback_success_rate']:.6f}`, fail `{row['failure_rate']:.6f}`.")
    warnings = []
    if not df_stage.empty and float(df_stage["ai_success_rate"].max()) <= 0.0:
        warnings.append("The learned AI stage contributed zero exact rescues in this run.")
    if not df_qhit.empty and not df_qhit[(df_qhit["decoder"] != "ldpc_only") & (df_qhit["query_cap_hit_rate"] >= 0.75)].empty:
        warnings.append("At least one rescue decoder hit its query cap on 75%+ of invoked frames, indicating search saturation.")
    if not df_div.empty and not df_div[(df_div["decoder"] == "ldpc_only") & (df_div["perfect_duplication_rate"] > 0.50)].empty:
        warnings.append("High worker duplication was detected; Monte Carlo streams may not be independent across workers.")
    if not df_gain_final_cap.empty:
        lines += ["", "## Budget-matched comparisons"]
        for ebn0_db, g in df_gain_final_cap.groupby("ebn0_db", sort=True):
            row = g[g["decoder"] == PRIMARY_AI_DECODER]
            if not row.empty:
                lines.append(f"- `Eb/N0={ebn0_db:.2f} dB`: Active-Set MiM minus `final_llr_grand_capmatched` = `{float(row.iloc[0]['net_success_gain_over_final_capmatched']):.6f}` net exact success.")
    if not df_gain_guard_best.empty:
        for ebn0_db, g in df_gain_guard_best.groupby("ebn0_db", sort=True):
            row = g[g["decoder"] == PRIMARY_AI_DECODER]
            if not row.empty:
                lines.append(f"- `Eb/N0={ebn0_db:.2f} dB`: Active-Set MiM minus `guard_plus_best_syndrome` = `{float(row.iloc[0]['net_success_gain_over_guard_plus_best_syndrome']):.6f}` net exact success.")
    if warnings:
        lines += ["", "## Warnings"]
        lines += [f"- {w}" for w in warnings]
    lines += ["", "## Publication files", "- `publication_main_table.csv`", "- `publication_delta_table.csv`", "- `summary_eval.csv`", "- `summary.md`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    for col in BOOL_COLS:
        if col in frame_df.columns:
            frame_df[col] = _normalize_boolish_series(frame_df[col])
    for col in ["ebn0_db", "queries", "solver_states", "frontier_peak", "pattern_weight", "selected_snapshot", "selected_syndrome_weight"]:
        if col in frame_df.columns:
            frame_df[col] = pd.to_numeric(frame_df[col], errors="coerce").fillna(0)

    frame_df = frame_df.sort_values(["ebn0_db", "decoder", "worker_id", "frame_idx"]).reset_index(drop=True)
    frame_df.to_csv(report_dir / "frame_rows_all.csv", index=False)

    quantiles = tuple(float(q) for q in cfg.get("report", {}).get("query_quantiles", [0.5, 0.9, 0.99]))
    df_summary = summarize_frames(frame_df, quantiles=quantiles).sort_values(["decoder", "ebn0_db"]).reset_index(drop=True)
    df_summary.to_csv(report_dir / "summary_eval.csv", index=False)

    df_gain = _gain_over_ldpc(df_summary)
    df_gain.to_csv(report_dir / "net_success_gain_over_ldpc.csv", index=False)
    df_gain_final = _gain_over_reference(df_summary, "final_llr_grand", "net_success_gain_over_final_llr")
    if not df_gain_final.empty:
        df_gain_final.to_csv(report_dir / "net_success_gain_over_final_llr.csv", index=False)
    df_gain_final_cap = _gain_over_reference(df_summary, "final_llr_grand_capmatched", "net_success_gain_over_final_capmatched")
    if not df_gain_final_cap.empty:
        df_gain_final_cap.to_csv(report_dir / "net_success_gain_over_final_capmatched.csv", index=False)
    df_gain_guard_best = _gain_over_reference(df_summary, "guard_plus_best_syndrome", "net_success_gain_over_guard_plus_best_syndrome")
    if not df_gain_guard_best.empty:
        df_gain_guard_best.to_csv(report_dir / "net_success_gain_over_guard_plus_best_syndrome.csv", index=False)
    df_gap = _gap_to_oracle(df_summary)
    df_gap.to_csv(report_dir / "gap_to_oracle_summary.csv", index=False)
    df_eff = _query_efficiency(df_gain, df_summary)
    df_eff.to_csv(report_dir / "query_efficiency_summary.csv", index=False)
    df_qhit = _query_cap_hit_rates(frame_df)
    df_qhit.to_csv(report_dir / "query_cap_hit_summary.csv", index=False)
    df_pusage = _primitive_usage(frame_df)
    df_pusage.to_csv(report_dir / "primitive_usage_summary.csv", index=False)
    df_stage = _ai_stage_contribution(frame_df)
    df_stage.to_csv(report_dir / "ai_stage_contribution_summary.csv", index=False)
    df_stage_visit = _ai_stage_visit(frame_df)
    df_stage_visit.to_csv(report_dir / "ai_stage_visit_summary.csv", index=False)
    df_div = _worker_diversity(frame_df)
    df_div.to_csv(report_dir / "worker_diversity_summary.csv", index=False)
    _publication_tables(df_summary, df_gain_final_cap, df_gain_guard_best, report_dir)

    _line_plot(df_summary, "ebn0_db", "net_exact_success_rate", report_dir / "net_exact_success_rate_vs_ebn0", "Net exact success rate")
    _line_plot(df_summary, "ebn0_db", "net_frame_error_rate", report_dir / "net_frame_error_rate_vs_ebn0", "Net frame error rate")
    _line_plot(df_summary, "ebn0_db", "conditional_exact_success_rate", report_dir / "conditional_rescue_success_rate_vs_ebn0", "Conditional rescue success rate")
    _line_plot(df_summary, "ebn0_db", "avg_queries_on_invoked_frames", report_dir / "avg_queries_on_invoked_frames_vs_ebn0", "Avg queries on invoked frames", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_queries_per_original_frame", report_dir / "avg_queries_per_original_frame_vs_ebn0", "Avg queries per original frame", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_solver_states_per_original_frame", report_dir / "avg_solver_states_per_original_frame_vs_ebn0", "Avg solver states per original frame", include_ldpc=False)
    _line_plot(df_gain, "ebn0_db", "net_success_gain_over_ldpc", report_dir / "net_success_gain_over_ldpc_vs_ebn0", "Net success gain over LDPC", include_ldpc=False)
    if not df_gain_final.empty:
        _line_plot(df_gain_final, "ebn0_db", "net_success_gain_over_final_llr", report_dir / "net_success_gain_over_final_llr_vs_ebn0", "Net success gain over final LLR GRAND", include_ldpc=False)
    if not df_gain_final_cap.empty:
        _line_plot(df_gain_final_cap, "ebn0_db", "net_success_gain_over_final_capmatched", report_dir / "net_success_gain_over_final_capmatched_vs_ebn0", "Net success gain over final cap-matched GRAND", include_ldpc=False)
    if not df_gain_guard_best.empty:
        _line_plot(df_gain_guard_best, "ebn0_db", "net_success_gain_over_guard_plus_best_syndrome", report_dir / "net_success_gain_over_guard_plus_best_syndrome_vs_ebn0", "Net success gain over guard+best-syndrome", include_ldpc=False)
    if not df_gap.empty:
        _line_plot(df_gap, "ebn0_db", "net_success_gap_to_oracle", report_dir / "net_gap_to_oracle_vs_ebn0", "Net success gap to oracle", include_ldpc=False)
        _line_plot(df_gap, "ebn0_db", "selected_snapshot_gap_to_oracle", report_dir / "snapshot_gap_to_oracle_vs_ebn0", "Snapshot gap to oracle", include_ldpc=False)
    if not df_eff.empty:
        _line_plot(df_eff, "ebn0_db", "net_success_gain_per_1000_queries", report_dir / "query_efficiency_vs_ebn0", "Net gain per 1000 queries", include_ldpc=False)
        _line_plot(df_eff, "ebn0_db", "net_success_gain_per_1000_solver_states", report_dir / "solver_efficiency_vs_ebn0", "Net gain per 1000 solver states", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_selected_snapshot", report_dir / "avg_selected_snapshot_vs_ebn0", "Avg selected snapshot")
    _line_plot(df_summary, "ebn0_db", "avg_frontier_peak", report_dir / "avg_frontier_peak_vs_ebn0", "Avg frontier peak", include_ldpc=False)
    if not df_qhit.empty:
        _line_plot(df_qhit, "ebn0_db", "query_cap_hit_rate", report_dir / "query_cap_hit_rate_vs_ebn0", "Query-budget hit rate", include_ldpc=False)
    if not df_stage.empty:
        _line_plot_multi(df_stage, "ebn0_db", ["guard_success_rate", "ai_success_rate", "fallback_success_rate", "failure_rate"], report_dir / "ai_stage_contribution_vs_ebn0", "Rate over invoked AI-rescue frames")
    if not df_stage_visit.empty:
        _line_plot_multi(df_stage_visit, "ebn0_db", ["guard_visit_rate", "ai_visit_rate", "fallback_visit_rate"], report_dir / "ai_stage_visit_vs_ebn0", "Stage visit rate")
    if not df_div.empty:
        _line_plot(df_div, "ebn0_db", "avg_unique_outcomes_per_frame_slot", report_dir / "worker_diversity_vs_ebn0", "Avg unique worker outcomes / frame slot")

    _write_markdown(df_summary, df_stage, df_stage_visit, df_div, df_qhit, report_dir, df_gain_final_cap, df_gain_guard_best)


if __name__ == "__main__":
    main()
