# Novel_GRAND — Active-Set MiM GRAND

This repository evaluates a hybrid **5G NR LDPC + GRAND rescue** pipeline on FIR using Sionna-generated 5G NR channels.

The current standalone package implements:

- legacy 5G NR LDPC decoding with early stop,
- a strong `final_llr_grand` guard on detected failures,
- a learned **snapshot selector**,
- a learned **bit-risk scorer** on the selected snapshot,
- an **exact meet-in-the-middle syndrome solver** on a compact AI-selected active set,
- a conservative `best_syndrome_llr` fallback,
- publication-quality report generation with both CSV and PDF/PNG plots.

## Quick start on FIR

```bash
cd /home/rsadve1/scratch/Novel_GRAND
source env/activate_fir.sh
python -m pip install -e .

bash tools/check_tags_package.sh
sbatch slurm/00_smoke_tags_grand.sbatch
bash slurm/submit_legacy_probe.sh
bash tools/check_pipeline_status.sh
bash slurm/submit_tags_grand_pipeline.sh
bash tools/check_pipeline_status.sh
```

## Main outputs

All generated outputs are placed under:

```text
outputs/fir_tags_grand_autotuned/
```

The most important artifacts are:

- `reports/summary_eval.csv`
- `reports/publication_main_table.csv`
- `reports/summary.md`
- publication plots in both `.png` and `.pdf`

## Core scientific question

Does the AI rescue path beat strong non-AI baselines under a fair budget?

The package reports this explicitly against:

- `final_llr_grand_capmatched`
- `guard_plus_best_syndrome`
- `oracle_best_llr` (upper bound)
