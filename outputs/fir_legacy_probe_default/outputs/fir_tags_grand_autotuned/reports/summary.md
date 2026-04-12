# TAGS-GRAND-Lite report

## Interpretation

For rescue decoders, **conditional** success means success only on frames that legacy LDPC had already failed. **Net** success folds those rescues back into the whole-frame success rate and is the correct metric for comparing against legacy LDPC.

## Best decoder by Eb/N0 (net exact success)

- `Eb/N0=11.00 dB`: **final_llr_grand** with net exact success `0.872000` and average queries per original frame `670.32`.
- `Eb/N0=12.00 dB`: **final_llr_grand** with net exact success `0.916000` and average queries per original frame `420.40`.
- `Eb/N0=13.00 dB`: **oracle_best_llr** with net exact success `0.963375` and average queries per original frame `185.05`.
- `Eb/N0=14.00 dB`: **oracle_best_llr** with net exact success `0.964063` and average queries per original frame `179.90`.

## Best conditional rescue decoder

- `Eb/N0=11.00 dB`: **final_llr_grand** conditional rescue success `0.200000`, queries on invoked frames `4189.48`.
- `Eb/N0=12.00 dB`: **final_llr_grand** conditional rescue success `0.045455`, queries on invoked frames `4777.27`.
- `Eb/N0=13.00 dB`: **oracle_best_llr** conditional rescue success `0.265664`, queries on invoked frames `3710.32`.
- `Eb/N0=14.00 dB`: **oracle_best_llr** conditional rescue success `0.108527`, queries on invoked frames `4462.54`.

## Generated files

- `frame_rows_all.csv`
- `summary_eval.csv`
- `best_decoder_by_ebn0.csv`
- `best_rescue_decoder_by_ebn0.csv`
- `net_exact_success_rate_vs_ebn0.png`
- `net_frame_error_rate_vs_ebn0.png`
- `conditional_rescue_success_rate_vs_ebn0.png`
- `avg_queries_on_invoked_frames_vs_ebn0.png`
- `avg_queries_per_original_frame_vs_ebn0.png`
- `net_success_gain_over_ldpc_vs_ebn0.png`
- `avg_selected_snapshot_vs_ebn0.png`
- `avg_frontier_peak_vs_ebn0.png`