# Novel_GRAND standalone package – update 3

This update addresses the latest full-run findings.

## Fixed
- `summary.md` is now regenerated correctly by the report stage. The stray `PLACEHOLDER` token that could leave Markdown reports stale is removed.
- `check_pipeline_status.sh` now warns when `summary.md` is older than `summary_eval.csv`, and when probe recommendation files disagree.

## Improved
- TAGS-GRAND-Lite now uses a **focused AI rescue schedule** instead of fragmenting the AI budget evenly across many snapshots.
- The snapshot chooser is biased toward earlier, lower-syndrome snapshots that empirically sit closer to the oracle-best rescue point.
- The focused AI stage uses a richer frontier (`ai_focus_topk_bits`, `ai_focus_expand_width`) while keeping the total budget compatible with the previous run.

## Added
- Two analysis baselines:
  - `selector_llr_grand`
  - `selector_blend_grand`
- New reports:
  - `gap_to_oracle_summary.csv`
  - `net_gap_to_oracle_vs_ebn0.png`
  - `snapshot_gap_to_oracle_vs_ebn0.png`
  - `query_efficiency_summary.csv`
  - `query_efficiency_vs_ebn0.png`
  - `tags_stage_visit_summary.csv`
  - `tags_stage_visit_vs_ebn0.png`

## Why this matters
The previous published run showed that Monte Carlo diversity had improved, but the AI rescue stage still contributed zero exact rescues while consuming about 2.5x the query budget of the strongest non-AI baseline. This update is designed to answer two questions cleanly on the next rerun:
1. Does AI snapshot selection alone add value?
2. Does a focused AI rescue stage recover any of the oracle headroom without exploding query cost?
