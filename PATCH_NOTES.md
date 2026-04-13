# Novel_GRAND standalone v6 budget-audit package

This package is a standalone follow-up to v5.

What changed:
- Added fair cap-matched non-AI baselines:
  - `final_llr_grand_capmatched`
  - `best_syndrome_llr_grand_capmatched`
  - `guard_plus_best_syndrome`
- Kept the current TAGS model unchanged so the next run isolates methodology from architecture.
- Added report outputs comparing TAGS against cap-matched baselines.
- Added an explicit markdown warning when TAGS is compared to a smaller-budget final-LLR baseline.
- Simplified the default evaluation set by dropping dead baselines that were not informative.

Why this update is needed:
The v5 results showed that TAGS only slightly beat `final_llr_grand`, but TAGS had a larger total query budget. This package makes the next run fairer and much more informative.
