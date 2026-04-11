# Novel_GRAND repair overlay

This overlay fixes the broken evaluation stage and adds a legacy-LDPC operating-point probe.

## What it changes

- fixes PyTorch checkpoint loading in `novel_grand/models/training.py`
- adds `novel_grand/scripts/probe_legacy.py`
- adds `slurm/05_probe_legacy_tags_grand.sbatch` and `slurm/submit_legacy_probe.sh`
- updates the Slurm scripts so a config path can be supplied through `NOVEL_GRAND_CONFIG`
- updates `configs/fir_default.yaml` to a more plausible rescue-region SNR ladder
- adds safe cleanup helpers under `tools/`

## Recommended usage

1. Run `bash tools/clean_tags_repo.sh`
2. Run `bash tools/ensure_gitignore_patterns.sh`
3. Run `sbatch slurm/00_smoke_tags_grand.sbatch`
4. Run `bash slurm/submit_legacy_probe.sh`
5. Inspect `outputs/fir_legacy_probe_default/reports/recommended_full_config.yaml`
6. Run `bash slurm/submit_tags_grand_pipeline.sh outputs/fir_legacy_probe_default/reports/recommended_full_config.yaml`
