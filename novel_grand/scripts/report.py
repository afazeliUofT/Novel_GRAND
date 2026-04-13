from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novel_grand.config import load_config, run_root
from novel_grand.sim.aggregate import load_frame_rows, summarize_frames


BOOL_COLS = ["legacy_detected_failure", "success_exact", "valid_codeword", "undetected_error"]



def _normalize_boolish_series(s: pd.Series) -> pd.Series:
    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
    }
    return s.map(mapping).fillna(0).astype(int)



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



def _line_plot_multi(df: pd.DataFrame, x: str, y_cols: list[str], out_path: Path, ylabel: str) -> None:
    plt.figure()
    gg = df.sort_values(x)
    for col in y_cols:
        if col not in gg.columns:
            continue
        plt.plot(gg[x], gg[col], marker="o", label=col)
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


def _gain_over_reference(df_summary: pd.DataFrame, ref_decoder: str, out_col: str) -> pd.DataFrame:
    rows = []
    for ebn0_db, g in df_summary.groupby("ebn0_db", sort=True):
        ref_row = g[g["decoder"] == ref_decoder]
        if ref_row.empty:
            continue
        ref_net = float(ref_row.iloc[0]["net_exact_success_rate"])
        ref_q = float(ref_row.iloc[0].get("avg_queries_per_original_frame", 0.0))
        for _, row in g.iterrows():
            rows.append(
                {
                    "ebn0_db": float(ebn0_db),
                    "decoder": str(row["decoder"]),
                    out_col: float(row["net_exact_success_rate"] - ref_net),
                    "extra_queries_vs_reference": float(row.get("avg_queries_per_original_frame", 0.0) - ref_q),
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
        df_summary[["ebn0_db", "decoder", "avg_queries_per_original_frame"]],
        on=["ebn0_db", "decoder"],
        how="left",
    )
    q = merged["avg_queries_per_original_frame"].replace(0.0, np.nan)
    merged["net_success_gain_per_1000_queries"] = 1000.0 * merged["net_success_gain_over_ldpc"] / q
    merged["net_success_gain_per_1000_queries"] = merged["net_success_gain_per_1000_queries"].fillna(0.0)
    return merged



def _decoder_budget(cfg: dict, decoder: str) -> int:
    if decoder == "ldpc_only":
        return 0
    q_main = int(cfg["grand"]["query_cap"])
    if decoder == "tags_grand_lite":
        q_rescue = int(cfg["grand"].get("rescue_bonus_cap", q_main))
        q_fb = int(cfg["grand"].get("fallback_bonus_cap", max(1000, q_main // 2)))
        return q_main + q_rescue + q_fb
    return q_main



def _query_cap_hit_rates(frame_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    if frame_df.empty:
        return pd.DataFrame()
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        budget = int(g["query_budget"].max()) if "query_budget" in g.columns else _decoder_budget(cfg, str(decoder))
        hit_rate = float((g["queries"] >= budget).mean()) if budget > 0 else 0.0
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "decoder": str(decoder),
                "query_budget": budget,
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
            tokens = [t.strip() for t in str(s).replace("|", ",").split(",") if t.strip()]
            token_set = set(tokens)
            bit += int("bit" in token_set)
            group += int("group" in token_set)
            guard += int("guard_final_llr" in token_set)
            fallback += int("fallback_best_syndrome_llr" in token_set)
            ai += int(any(tok.startswith("ai_") for tok in token_set))
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



PRIMARY_AI_DECODER = "flowsearch_grand"


def _ai_stage_contribution(frame_df: pd.DataFrame) -> pd.DataFrame:
    g_all = frame_df[frame_df["decoder"] == PRIMARY_AI_DECODER].copy()
    if g_all.empty:
        return pd.DataFrame()
    rows = []
    for ebn0_db, g in g_all.groupby("ebn0_db", sort=True):
        n = len(g)
        kinds = g["primitive_kinds"].fillna("").astype(str)
        success = g["success_exact"].astype(bool)
        guard_success = (success & kinds.str.contains("guard_final_llr") & ~kinds.str.contains("ai_") & ~kinds.str.contains("fallback_best_syndrome_llr")).sum()
        ai_success = (success & kinds.str.contains("ai_") & ~kinds.str.contains("fallback_best_syndrome_llr")).sum()
        fallback_success = (success & kinds.str.contains("fallback_best_syndrome_llr")).sum()
        fail = n - (guard_success + ai_success + fallback_success)
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "guard_success_rate": guard_success / max(n, 1),
                "ai_success_rate": ai_success / max(n, 1),
                "fallback_success_rate": fallback_success / max(n, 1),
                "failure_rate": fail / max(n, 1),
            }
        )
    return pd.DataFrame(rows)



def _ai_stage_visit(frame_df: pd.DataFrame) -> pd.DataFrame:
    g_all = frame_df[frame_df["decoder"] == PRIMARY_AI_DECODER].copy()
    if g_all.empty:
        return pd.DataFrame()
    rows = []
    for ebn0_db, g in g_all.groupby("ebn0_db", sort=True):
        n = len(g)
        kinds = g["primitive_kinds"].fillna("").astype(str)
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "guard_visit_rate": float(kinds.str.contains("guard_final_llr").mean()),
                "ai_visit_rate": float(kinds.str.contains("ai_").mean()),
                "fallback_visit_rate": float(kinds.str.contains("fallback_best_syndrome_llr").mean()),
                "n_frames": int(n),
            }
        )
    return pd.DataFrame(rows)



def _worker_diversity(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()
    rows = []
    sig_cols = [
        "legacy_detected_failure",
        "selected_snapshot",
        "selected_syndrome_weight",
        "queries",
        "frontier_peak",
        "pattern_weight",
        "success_exact",
        "valid_codeword",
        "undetected_error",
    ]
    for (ebn0_db, decoder), g in frame_df.groupby(["ebn0_db", "decoder"], sort=True):
        unique_counts = []
        for _, gf in g.groupby("frame_idx", sort=True):
            sig = gf[sig_cols].astype(str).agg("|".join, axis=1)
            unique_counts.append(sig.nunique())
        unique_counts = np.asarray(unique_counts, dtype=float)
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "decoder": str(decoder),
                "avg_unique_outcomes_per_frame_slot": float(unique_counts.mean()) if unique_counts.size else 0.0,
                "perfect_duplication_rate": float(np.mean(unique_counts <= 1.0 + 1e-12)) if unique_counts.size else 0.0,
                "max_unique_outcomes_per_frame_slot": float(unique_counts.max()) if unique_counts.size else 0.0,
                "n_frame_slots": int(unique_counts.size),
            }
        )
    return pd.DataFrame(rows)



def _write_markdown(
    df_summary: pd.DataFrame,
    df_stage: pd.DataFrame,
    df_stage_visit: pd.DataFrame,
    df_div: pd.DataFrame,
    df_qhit: pd.DataFrame,
    out_dir: Path,
    df_gain_final_cap: pd.DataFrame,
    df_gain_guard_best: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# FlowSearch-GRAND report")
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
    if not df_stage.empty:
        lines.append("")
        lines.append("## AI stage contribution (conditional on LDPC-detected failures)")
        for _, row in df_stage.sort_values("ebn0_db").iterrows():
            lines.append(
                f"- `Eb/N0={row['ebn0_db']:.2f} dB`: guard `{row['guard_success_rate']:.6f}`, ai `{row['ai_success_rate']:.6f}`, fallback `{row['fallback_success_rate']:.6f}`, fail `{row['failure_rate']:.6f}`."
            )
    if not df_stage_visit.empty:
        lines.append("")
        lines.append("## AI stage visit rates")
        for _, row in df_stage_visit.sort_values("ebn0_db").iterrows():
            lines.append(
                f"- `Eb/N0={row['ebn0_db']:.2f} dB`: guard visited `{row['guard_visit_rate']:.6f}`, ai visited `{row['ai_visit_rate']:.6f}`, fallback visited `{row['fallback_visit_rate']:.6f}`."
            )
    warnings = []
    if not df_div.empty:
        warn_dup = df_div[(df_div["decoder"] == "ldpc_only") & (df_div["perfect_duplication_rate"] > 0.50)]
        if not warn_dup.empty:
            warnings.append("High worker duplication was detected. Monte Carlo streams may not be independent across workers.")
    if not df_stage.empty and float(df_stage["ai_success_rate"].max()) <= 0.0:
        warnings.append("The learned AI rescue stage contributed zero exact rescues in this run.")
    if not df_qhit.empty:
        hi = df_qhit[(df_qhit["decoder"] != "ldpc_only") & (df_qhit["query_cap_hit_rate"] >= 0.75)]
        if not hi.empty:
            warnings.append("At least one rescue decoder hit its query cap on 75%+ of invoked frames, indicating search saturation.")
        tags_budget = df_qhit[df_qhit["decoder"] == "tags_grand_lite"]["query_budget"]
        final_budget = df_qhit[df_qhit["decoder"] == "final_llr_grand"]["query_budget"]
        if not tags_budget.empty and not final_budget.empty and float(tags_budget.iloc[0]) > float(final_budget.iloc[0]):
            warnings.append("Compare the AI decoder against the cap-matched and guard-plus baselines before attributing small gains to learned ordering.")
    if not df_gain_final_cap.empty:
        lines.append("")
        lines.append("## Budget-matched comparisons")
        for ebn0_db, g in df_gain_final_cap.groupby("ebn0_db", sort=True):
            row = g[g["decoder"] == PRIMARY_AI_DECODER]
            if not row.empty:
                gain = float(row.iloc[0]["net_success_gain_over_final_capmatched"])
                lines.append(f"- `Eb/N0={ebn0_db:.2f} dB`: FlowSearch minus `final_llr_grand_capmatched` = `{gain:.6f}` net exact success.")
    if not df_gain_guard_best.empty:
        for ebn0_db, g in df_gain_guard_best.groupby("ebn0_db", sort=True):
            row = g[g["decoder"] == PRIMARY_AI_DECODER]
            if not row.empty:
                gain = float(row.iloc[0]["net_success_gain_over_guard_plus_best_syndrome"])
                lines.append(f"- `Eb/N0={ebn0_db:.2f} dB`: FlowSearch minus `guard_plus_best_syndrome` = `{gain:.6f}` net exact success.")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Files")
    for name in [
        "summary_eval.csv",
        "summary.md",
        "frame_rows_all.csv",
        "net_exact_success_rate_vs_ebn0.png",
        "net_frame_error_rate_vs_ebn0.png",
        "conditional_rescue_success_rate_vs_ebn0.png",
        "avg_queries_on_invoked_frames_vs_ebn0.png",
        "avg_queries_per_original_frame_vs_ebn0.png",
        "net_success_gain_over_ldpc.csv",
        "net_success_gain_over_ldpc_vs_ebn0.png",
        "net_success_gain_over_final_llr.csv",
        "net_success_gain_over_final_llr_vs_ebn0.png",
        "net_success_gain_over_final_capmatched.csv",
        "net_success_gain_over_final_capmatched_vs_ebn0.png",
        "net_success_gain_over_guard_plus_best_syndrome.csv",
        "net_success_gain_over_guard_plus_best_syndrome_vs_ebn0.png",
        "gap_to_oracle_summary.csv",
        "net_gap_to_oracle_vs_ebn0.png",
        "snapshot_gap_to_oracle_vs_ebn0.png",
        "query_efficiency_summary.csv",
        "query_efficiency_vs_ebn0.png",
        "avg_selected_snapshot_vs_ebn0.png",
        "avg_frontier_peak_vs_ebn0.png",
        "query_cap_hit_rate_vs_ebn0.png",
        "primitive_usage_summary.csv",
        "query_cap_hit_summary.csv",
        "ai_stage_contribution_summary.csv",
        "ai_stage_contribution_vs_ebn0.png",
        "ai_stage_visit_summary.csv",
        "ai_stage_visit_vs_ebn0.png",
        "worker_diversity_summary.csv",
        "worker_diversity_vs_ebn0.png",
    ]:
        lines.append(f"- `{name}`")

    out_path = out_dir / "summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



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
    for col in ["ebn0_db", "queries", "frontier_peak", "pattern_weight", "selected_snapshot", "selected_syndrome_weight"]:
        if col in frame_df.columns:
            frame_df[col] = pd.to_numeric(frame_df[col], errors="coerce").fillna(0)

    frame_df = frame_df.sort_values(["ebn0_db", "decoder", "worker_id", "frame_idx"]).reset_index(drop=True)
    frame_df.to_csv(report_dir / "frame_rows_all.csv", index=False)

    quantiles = tuple(float(q) for q in cfg.get("report", {}).get("query_quantiles", [0.5, 0.9, 0.99]))
    df_summary = summarize_frames(frame_df, quantiles=quantiles)
    df_summary = df_summary.sort_values(["decoder", "ebn0_db"]).reset_index(drop=True)
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
    df_qhit = _query_cap_hit_rates(frame_df, cfg)
    df_qhit.to_csv(report_dir / "query_cap_hit_summary.csv", index=False)
    df_pusage = _primitive_usage(frame_df)
    df_pusage.to_csv(report_dir / "primitive_usage_summary.csv", index=False)
    df_stage = _ai_stage_contribution(frame_df)
    df_stage.to_csv(report_dir / "ai_stage_contribution_summary.csv", index=False)
    df_stage_visit = _ai_stage_visit(frame_df)
    df_stage_visit.to_csv(report_dir / "ai_stage_visit_summary.csv", index=False)
    df_div = _worker_diversity(frame_df)
    df_div.to_csv(report_dir / "worker_diversity_summary.csv", index=False)

    _line_plot(df_summary, "ebn0_db", "net_exact_success_rate", report_dir / "net_exact_success_rate_vs_ebn0.png", "Net exact success rate")
    _line_plot(df_summary, "ebn0_db", "net_frame_error_rate", report_dir / "net_frame_error_rate_vs_ebn0.png", "Net frame error rate")
    _line_plot(df_summary, "ebn0_db", "conditional_exact_success_rate", report_dir / "conditional_rescue_success_rate_vs_ebn0.png", "Conditional rescue success rate")
    _line_plot(df_summary, "ebn0_db", "avg_queries_on_invoked_frames", report_dir / "avg_queries_on_invoked_frames_vs_ebn0.png", "Avg queries on invoked frames", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_queries_per_original_frame", report_dir / "avg_queries_per_original_frame_vs_ebn0.png", "Avg queries per original frame", include_ldpc=False)
    _line_plot(df_gain, "ebn0_db", "net_success_gain_over_ldpc", report_dir / "net_success_gain_over_ldpc_vs_ebn0.png", "Net success gain over LDPC", include_ldpc=False)
    if not df_gain_final.empty:
        _line_plot(df_gain_final, "ebn0_db", "net_success_gain_over_final_llr", report_dir / "net_success_gain_over_final_llr_vs_ebn0.png", "Net success gain over final LLR GRAND", include_ldpc=False)
    if not df_gain_final_cap.empty:
        _line_plot(df_gain_final_cap, "ebn0_db", "net_success_gain_over_final_capmatched", report_dir / "net_success_gain_over_final_capmatched_vs_ebn0.png", "Net success gain over final cap-matched GRAND", include_ldpc=False)
    if not df_gain_guard_best.empty:
        _line_plot(df_gain_guard_best, "ebn0_db", "net_success_gain_over_guard_plus_best_syndrome", report_dir / "net_success_gain_over_guard_plus_best_syndrome_vs_ebn0.png", "Net success gain over non-AI guard+best-syndrome", include_ldpc=False)
    if not df_gap.empty:
        _line_plot(df_gap, "ebn0_db", "net_success_gap_to_oracle", report_dir / "net_gap_to_oracle_vs_ebn0.png", "Net success gap to oracle", include_ldpc=False)
        _line_plot(df_gap, "ebn0_db", "selected_snapshot_gap_to_oracle", report_dir / "snapshot_gap_to_oracle_vs_ebn0.png", "Snapshot gap to oracle", include_ldpc=False)
    if not df_eff.empty:
        _line_plot(df_eff, "ebn0_db", "net_success_gain_per_1000_queries", report_dir / "query_efficiency_vs_ebn0.png", "Net gain per 1000 queries", include_ldpc=False)
    _line_plot(df_summary, "ebn0_db", "avg_selected_snapshot", report_dir / "avg_selected_snapshot_vs_ebn0.png", "Avg selected snapshot")
    _line_plot(df_summary, "ebn0_db", "avg_frontier_peak", report_dir / "avg_frontier_peak_vs_ebn0.png", "Avg frontier peak", include_ldpc=False)
    if not df_qhit.empty:
        _line_plot(df_qhit, "ebn0_db", "query_cap_hit_rate", report_dir / "query_cap_hit_rate_vs_ebn0.png", "Query-budget hit rate", include_ldpc=False)
    if not df_stage.empty:
        _line_plot_multi(df_stage, "ebn0_db", ["guard_success_rate", "ai_success_rate", "fallback_success_rate", "failure_rate"], report_dir / "ai_stage_contribution_vs_ebn0.png", "Rate over invoked AI-rescue frames")
    if not df_stage_visit.empty:
        _line_plot_multi(df_stage_visit, "ebn0_db", ["guard_visit_rate", "ai_visit_rate", "fallback_visit_rate"], report_dir / "ai_stage_visit_vs_ebn0.png", "Stage visit rate")
    if not df_div.empty:
        _line_plot(df_div, "ebn0_db", "avg_unique_outcomes_per_frame_slot", report_dir / "worker_diversity_vs_ebn0.png", "Avg unique worker outcomes / frame slot")

    _write_markdown(df_summary, df_stage, df_stage_visit, df_div, df_qhit, report_dir, df_gain_final_cap, df_gain_guard_best)


if __name__ == "__main__":
    main()
