# Patch notes: standalone v8 MaskDiff-GRAND package

This standalone package replaces the previous FlowSearch-GRAND AI stage with a new **MaskDiff-GRAND** rescue path.

## Main changes

- Replaced the multi-snapshot policy-value tree search with a **verifier-guided masked-diffusion set generator**.
- Kept the **teacher-aligned snapshot selector**, because the previous run showed that the selector was useful even though the search stage was weak.
- Trains the AI only on **post-guard failures**, i.e. the distribution the AI stage actually sees at inference.
- Generates **whole correction-set hypotheses** on a shortlist of risky bits, instead of stepwise local actions.
- Adds a lightweight **consensus reconditioning** pass and a local exact-repair step.
- Keeps **budget-matched baselines** so the next run can decide clearly whether the new AI stage is actually useful.

## Why this update exists

The previous package showed:

- the pipeline was mechanically sound,
- the selected snapshot moved much closer to oracle,
- but the AI stage itself still contributed almost nothing,
- and the strongest non-AI two-stage baseline remained hard to beat.

This package therefore changes the **AI rescue architecture itself**, not just the reporting or scheduling.
