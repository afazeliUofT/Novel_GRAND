# Novel_GRAND standalone v11 — Active-Set MiM GRAND

This package replaces the weak AI generator path with a solver-augmented rescue decoder.

## Main idea

After the strong `final_llr_grand` guard, the AI is used only where the previous experiments showed signal:

1. select the best LDPC snapshot;
2. score a compact active set of risky bits on that snapshot;
3. solve the residual syndrome equation **exactly** inside that active set with a meet-in-the-middle solver;
4. use a conservative `best_syndrome_llr` fallback only if the active-set candidate is too ambiguous or fails.

## Why this is different

Earlier packages tried to have AI generate correction patterns directly.
The repo results showed that:

- the snapshot selector was consistently useful;
- the downstream AI generator/search stage contributed very little;
- cap-matched non-AI baselines remained hard to beat.

This version therefore moves to a **verifier-guided neuro-symbolic solver**:
AI proposes the *where*, and an exact solver handles the *what*.

## Outputs added for publication-quality reporting

The report stage now writes both **PNG** and **PDF** plots plus compact CSV tables, including:

- `publication_main_table.csv`
- `net_exact_success_rate_vs_ebn0.{png,pdf}`
- `net_success_gain_over_final_capmatched_vs_ebn0.{png,pdf}`
- `net_success_gain_over_guard_plus_best_syndrome_vs_ebn0.{png,pdf}`
- `query_efficiency_vs_ebn0.{png,pdf}`
- `solver_efficiency_vs_ebn0.{png,pdf}`
- `ai_stage_contribution_vs_ebn0.{png,pdf}`
- `worker_diversity_vs_ebn0.{png,pdf}`

## Runtime philosophy

Walltimes stay close to the observed FIR runtimes:

- smoke: 10 min
- probe: 15 min
- collect: 25 min
- train: 20 min
- eval: 25 min
- report: 10 min
