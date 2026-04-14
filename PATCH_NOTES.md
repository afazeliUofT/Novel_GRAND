# Novel_GRAND standalone v9: GFlow-TTA GRAND

This standalone package replaces the weak `maskdiff_grand` AI rescue path with
`gflowtta_grand`, a **GFlow-inspired, test-time-adapted** exact-verifier rescue
mechanism.

## Why this update

The pushed v8 results showed:

- `maskdiff_grand` only marginally beat `final_llr_grand_capmatched`.
- It did **not** beat `guard_plus_best_syndrome` consistently.
- The AI stage contributed essentially no exact rescues.
- The snapshot selector remained useful, but the downstream AI rescue was not.

## What changed

- Added `novel_grand/grand/gflowtta.py`
- Reused the existing snapshot selector and policy/value training rows in
  `novel_grand/ldpc/pv_features.py`
- Replaced masked-diffusion training with:
  - `snapshot_selector.pt`
  - `action_prior.pt`
  - `state_value.pt`
- The new AI stage now does:
  1. final-LLR guard
  2. learned snapshot selection
  3. diverse constructive candidate generation on the selected snapshot
  4. cheap test-time adaptation via consensus bias
  5. exact local repair + exact syndrome verification
  6. conservative best-syndrome fallback

## Metrics to inspect after the run

- `summary_eval.csv`
- `net_success_gain_over_final_capmatched.csv`
- `net_success_gain_over_guard_plus_best_syndrome.csv`
- `ai_stage_contribution_summary.csv`
- `query_efficiency_summary.csv`
- `worker_diversity_summary.csv`

## Success criterion for v9

`gflowtta_grand` should beat both:

- `final_llr_grand_capmatched`
- `guard_plus_best_syndrome`

by a **clearly larger** margin than v8, with a materially nonzero AI-stage
contribution.
