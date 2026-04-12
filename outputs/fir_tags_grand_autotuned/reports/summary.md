# TAGS-GRAND report

## Interpretation
For rescue decoders, **conditional** success means success only on frames that legacy LDPC had already failed.
**Net** success folds those rescues back into the whole-frame success rate and is the correct metric for comparing against legacy LDPC.

## Best decoder by Eb/N0 (net exact success)
- `Eb/N0=11.00 dB`: **oracle_best_llr** with net exact success `0.876000` and average queries per original frame `620.04`.
- `Eb/N0=12.00 dB`: **oracle_best_llr** with net exact success `0.948000` and average queries per original frame `262.95`.
- `Eb/N0=13.00 dB`: **oracle_best_llr** with net exact success `0.959187` and average queries per original frame `204.12`.
- `Eb/N0=14.00 dB`: **final_llr_grand** with net exact success `0.944375` and average queries per original frame `279.46`.

## Files
- `summary_eval.csv`
- `frame_rows_all.csv`
- `net_exact_success_rate_vs_ebn0.png`
- `net_frame_error_rate_vs_ebn0.png`
- `conditional_rescue_success_rate_vs_ebn0.png`
- `avg_queries_on_invoked_frames_vs_ebn0.png`
- `avg_queries_per_original_frame_vs_ebn0.png`
- `net_success_gain_over_ldpc_vs_ebn0.png`
- `avg_selected_snapshot_vs_ebn0.png`
- `avg_frontier_peak_vs_ebn0.png`
- `query_cap_hit_rate_vs_ebn0.png`
- `primitive_usage_summary.csv`
- `query_cap_hit_summary.csv`