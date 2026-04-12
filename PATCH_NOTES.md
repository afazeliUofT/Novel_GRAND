Novel_GRAND standalone v4 (teacher-fix)
======================================

What changed
------------
1. Teacher-aligned training targets
   - Snapshot labels are now based on the best LLR rescue snapshot under the
     actual GRAND search budget used during data collection.
   - Bit labels are now based on the teacher rescue pattern itself instead of
     raw wrong-bit masks on the Hamming-oracle snapshot.

2. TAGS-GRAND-Lite rescue flow
   - Guard stage remains final-iteration LLR GRAND.
   - AI stage is now a learned snapshot selector followed by LLR-ordered GRAND
     on the selected snapshot.
   - A learned blend search on the same selected snapshot is retained as a
     secondary AI stage.
   - Conservative best-syndrome fallback remains last.

3. Training / runtime defaults
   - Training epochs increased to 20.
   - Slightly larger MLPs.
   - Rescue/fallback budgets rebalanced to reduce wasted queries.

Why this update was needed
--------------------------
The previous v3 run showed:
- end-to-end pipeline worked,
- worker diversity looked reasonable,
- but the AI stage contributed zero exact rescues,
- and almost all TAGS gains came from the non-AI guard and tiny fallback.

The main issue was target mismatch: models were trained against raw Hamming-
distance or wrong-bit labels, while the deployment objective is exact rescue
within a finite query budget.
