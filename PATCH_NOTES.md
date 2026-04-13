# Novel_GRAND standalone v5 package

This package updates the v4 teacher-fix run based on the latest public results.

## Why this update exists

The v4 run showed that the mechanical pipeline works, but the novel AI stage is still weak:

- TAGS only slightly beats `final_llr_grand` in net exact success.
- The AI stage contributes only a handful of exact rescues.
- Query efficiency versus `final_llr_grand` is poor.
- The probe-generated config was not inheriting the intended v4 full-run hyperparameters.

## Main changes

1. **Probe-to-pipeline config consistency fixed**
   - `configs/fir_legacy_probe.yaml` now points to `configs/fir_default.yaml` as the base full-run config.
   - `recommended_full_config.yaml` is generated from that base config, not from the probe config itself.

2. **AI training aligned to the deployed stage**
   - Training data collection now keeps only **post-guard** LDPC failures for AI supervision.
   - Teacher snapshot search uses an **AI-stage-sized query cap** via `grand.ai_teacher_cap`.

3. **Main TAGS path focuses on the promising AI component**
   - The current results show that snapshot selection is the stronger AI contribution, while the learned blend stage is weak.
   - The default config now puts the full rescue budget into the **AI-selected snapshot + LLR-ordered GRAND** stage.
   - The learned blend path remains available for analysis baselines, but it is no longer the default main rescue budget sink.

4. **Safer score blending for learned bit scores**
   - The bit model no longer dominates the blended score by default.
   - LLR / channel / unsatisfied-check / flip signals are weighted more conservatively.

5. **Training defaults modestly strengthened**
   - More training frames per worker per SNR.
   - Larger hidden layers and more epochs in the main full-run config.

6. **New report artifact**
   - `net_success_gain_over_final_llr.csv`
   - `net_success_gain_over_final_llr_vs_ebn0.png`

7. **Reduced Slurm walltimes**
   - The previous limits were intentionally conservative during pipeline stabilization.
   - New limits better match the observed FIR runtimes.
