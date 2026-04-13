# MaskDiff-GRAND report

## Interpretation
For rescue decoders, **conditional** success means success only on frames that legacy LDPC had already failed.
**Net** success folds those rescues back into the whole-frame success rate and is the correct metric for comparing against legacy LDPC.

## Best decoder by Eb/N0 (net exact success)
- `Eb/N0=10.00 dB`: **oracle_best_llr** with net exact success `0.837812` and average queries per original frame `816.34`.
- `Eb/N0=11.00 dB`: **oracle_best_llr** with net exact success `0.894813` and average queries per original frame `530.70`.
- `Eb/N0=12.00 dB`: **oracle_best_llr** with net exact success `0.924562` and average queries per original frame `379.54`.
- `Eb/N0=13.00 dB`: **oracle_best_llr** with net exact success `0.948500` and average queries per original frame `259.50`.
- `Eb/N0=14.00 dB`: **oracle_best_llr** with net exact success `0.963938` and average queries per original frame `182.28`.

## AI stage contribution (conditional on LDPC-detected failures)
- `Eb/N0=10.00 dB`: guard `0.150224`, ai `0.000641`, fallback `0.007687`, fail `0.841448`.
- `Eb/N0=11.00 dB`: guard `0.135840`, ai `0.000000`, fallback `0.006516`, fail `0.857644`.
- `Eb/N0=12.00 dB`: guard `0.145251`, ai `0.000000`, fallback `0.004190`, fail `0.850559`.
- `Eb/N0=13.00 dB`: guard `0.169458`, ai `0.000000`, fallback `0.006897`, fail `0.823645`.
- `Eb/N0=14.00 dB`: guard `0.226018`, ai `0.000000`, fallback `0.005256`, fail `0.768725`.

## AI stage visit rates
- `Eb/N0=10.00 dB`: guard visited `1.000000`, ai visited `0.849776`, fallback visited `0.849135`.
- `Eb/N0=11.00 dB`: guard visited `1.000000`, ai visited `0.864160`, fallback visited `0.864160`.
- `Eb/N0=12.00 dB`: guard visited `1.000000`, ai visited `0.854749`, fallback visited `0.854749`.
- `Eb/N0=13.00 dB`: guard visited `1.000000`, ai visited `0.830542`, fallback visited `0.830542`.
- `Eb/N0=14.00 dB`: guard visited `1.000000`, ai visited `0.773982`, fallback visited `0.773982`.

## Budget-matched comparisons
- `Eb/N0=10.00 dB`: MaskDiff minus `final_llr_grand_capmatched` = `0.001375` net exact success.
- `Eb/N0=11.00 dB`: MaskDiff minus `final_llr_grand_capmatched` = `0.000437` net exact success.
- `Eb/N0=12.00 dB`: MaskDiff minus `final_llr_grand_capmatched` = `0.000250` net exact success.
- `Eb/N0=13.00 dB`: MaskDiff minus `final_llr_grand_capmatched` = `0.000312` net exact success.
- `Eb/N0=14.00 dB`: MaskDiff minus `final_llr_grand_capmatched` = `0.000250` net exact success.
- `Eb/N0=10.00 dB`: MaskDiff minus `guard_plus_best_syndrome` = `-0.000063` net exact success.
- `Eb/N0=11.00 dB`: MaskDiff minus `guard_plus_best_syndrome` = `-0.000250` net exact success.
- `Eb/N0=12.00 dB`: MaskDiff minus `guard_plus_best_syndrome` = `-0.000062` net exact success.
- `Eb/N0=13.00 dB`: MaskDiff minus `guard_plus_best_syndrome` = `0.000000` net exact success.
- `Eb/N0=14.00 dB`: MaskDiff minus `guard_plus_best_syndrome` = `-0.000188` net exact success.

## Warnings
- At least one rescue decoder hit its query cap on 75%+ of invoked frames, indicating search saturation.
- Compare the AI decoder against the cap-matched and guard-plus baselines before attributing small gains to learned ordering.

## Files
- `summary_eval.csv`
- `summary.md`
- `frame_rows_all.csv`
- `net_exact_success_rate_vs_ebn0.png`
- `net_frame_error_rate_vs_ebn0.png`
- `conditional_rescue_success_rate_vs_ebn0.png`
- `avg_queries_on_invoked_frames_vs_ebn0.png`
- `avg_queries_per_original_frame_vs_ebn0.png`
- `net_success_gain_over_ldpc.csv`
- `net_success_gain_over_ldpc_vs_ebn0.png`
- `net_success_gain_over_final_llr.csv`
- `net_success_gain_over_final_llr_vs_ebn0.png`
- `net_success_gain_over_final_capmatched.csv`
- `net_success_gain_over_final_capmatched_vs_ebn0.png`
- `net_success_gain_over_guard_plus_best_syndrome.csv`
- `net_success_gain_over_guard_plus_best_syndrome_vs_ebn0.png`
- `gap_to_oracle_summary.csv`
- `net_gap_to_oracle_vs_ebn0.png`
- `snapshot_gap_to_oracle_vs_ebn0.png`
- `query_efficiency_summary.csv`
- `query_efficiency_vs_ebn0.png`
- `avg_selected_snapshot_vs_ebn0.png`
- `avg_frontier_peak_vs_ebn0.png`
- `query_cap_hit_rate_vs_ebn0.png`
- `primitive_usage_summary.csv`
- `query_cap_hit_summary.csv`
- `ai_stage_contribution_summary.csv`
- `ai_stage_contribution_vs_ebn0.png`
- `ai_stage_visit_summary.csv`
- `ai_stage_visit_vs_ebn0.png`
- `worker_diversity_summary.csv`
- `worker_diversity_vs_ebn0.png`
