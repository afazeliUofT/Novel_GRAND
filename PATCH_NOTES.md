# Novel_GRAND standalone package – update 2

This update fixes and extends the standalone package used on FIR.

## Fixed
- Worker-to-worker Monte Carlo duplication by propagating unique seeds into Sionna PHY.
- `frame_rows_all.csv` mixed boolean/string columns are normalized before reporting.
- TAGS stage primitive logging now preserves the underlying primitive kinds from guard and fallback stages.
- Query-budget hit rate is computed against configured budgets instead of inferred maxima.

## Added
- `worker_diversity_summary.csv` and `worker_diversity_vs_ebn0.png` to detect duplicated worker streams.
- `tags_stage_contribution_summary.csv` and `tags_stage_contribution_vs_ebn0.png` to show whether guard, AI, or fallback is actually producing rescues.
- `query_budget` is now stored in frame rows for clearer diagnostics.

## Why this matters
The previous published run showed strong signs that many workers were simulating identical frame streams, which can make the effective sample size far smaller than the nominal `64 x frames_per_worker` count. This update is designed so the next rerun can be trusted scientifically.
