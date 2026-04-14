# Novel_GRAND standalone v10: MEMO-TTA GRAND

This standalone package replaces the weak `gflowtta_grand` AI rescue path with
`memotta_grand`, a **memory-augmented, test-time-adapted** exact-verifier rescue
mechanism.

## Why this update

The pushed v9 results showed:

- `gflowtta_grand` only marginally beat `final_llr_grand_capmatched`.
- It did **not** beat `guard_plus_best_syndrome` consistently.
- The AI stage contributed essentially no exact rescues.
- The snapshot selector remained useful, but the downstream AI rescue was not.

## What changed

- Added `novel_grand/grand/memotta.py`
- Reused the useful snapshot selector
- Replaced weak action generation with a persistent memory bank of successful
  post-guard rescue templates
- Added a learned `template_ranker.pt`
- The new AI stage now does:
  1. final-LLR guard
  2. learned snapshot selection
  3. retrieve similar successful rescue templates
  4. cheap test-time adaptation of template scores
  5. exact local repair + exact syndrome verification
  6. conservative best-syndrome fallback

## Metrics to inspect after the run

- `summary_eval.csv`
- `net_success_gain_over_final_capmatched.csv`
- `net_success_gain_over_guard_plus_best_syndrome.csv`
- `ai_stage_contribution_summary.csv`
- `query_efficiency_summary.csv`
- `worker_diversity_summary.csv`

## Success criterion for v10

`memotta_grand` should beat both:

- `final_llr_grand_capmatched`
- `guard_plus_best_syndrome`

by a **clearly larger** margin than v9, with a materially nonzero AI-stage
contribution.
