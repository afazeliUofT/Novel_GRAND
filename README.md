# Novel_GRAND

Standalone CPU package for FIR experiments on **5G NR LDPC + AI-aided GRAND rescue**.

This release uses **MEMO-TTA GRAND** as the main AI rescue path:

- legacy 5G NR LDPC first
- strong final-LLR GRAND guard on detected failures
- learned snapshot selector
- memory-augmented retrieval of successful post-guard rescue templates
- lightweight test-time adaptation of template scores for the current failed frame
- exact local repair and exact syndrome verification
- conservative best-syndrome fallback

The package is designed for the following workflow:

1. preserve `.git`, `.gitignore`, `.venv-fir`
2. wipe the rest of the repo directory
3. unzip this package
4. reinstall editable into the existing venv
5. run smoke → probe → full pipeline
6. push compact reports and plots to GitHub

## Main scripts

- `bash tools/check_tags_package.sh`
- `sbatch slurm/00_smoke_tags_grand.sbatch`
- `bash slurm/submit_legacy_probe.sh`
- `bash tools/check_pipeline_status.sh`
- `bash slurm/submit_tags_grand_pipeline.sh`

## Main output directories

- `outputs/fir_tags_grand_smoke/`
- `outputs/fir_legacy_probe_default/`
- `outputs/fir_tags_grand_autotuned/`

## Main report files

- `reports/summary_eval.csv`
- `reports/net_success_gain_over_final_capmatched.csv`
- `reports/net_success_gain_over_guard_plus_best_syndrome.csv`
- `reports/ai_stage_contribution_summary.csv`
- `reports/query_efficiency_summary.csv`
- `reports/worker_diversity_summary.csv`
