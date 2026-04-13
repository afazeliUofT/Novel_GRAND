# Novel GRAND standalone v7: FlowSearch replacement

This package replaces the previous static TAGS AI path with a new **FlowSearch-GRAND** rescue path.

## Main changes

- Replaces the earlier blend-based AI rescue with a **multi-snapshot policy-value tree search**.
- Keeps the strongest working deterministic pieces:
  - final-LLR GRAND guard
  - best-syndrome LLR fallback
- Adds new trainable models:
  - `snapshot_selector.pt`
  - `action_prior.pt`
  - `state_value.pt`
- Collects new training shards:
  - `action_rows_*.npz`
  - `value_rows_*.npz`
- Adds budget-fair reporting against:
  - `final_llr_grand_capmatched`
  - `guard_plus_best_syndrome`
- Updates Slurm walltimes to match observed FIR runtimes more closely.

## Why this version exists

The v6 results showed that the pipeline was scientifically credible, but the learned TAGS stage contributed only a tiny fraction of total rescues once the comparison was made budget-fair. This version tests a **different AI formulation** rather than another small patch of the same one.
