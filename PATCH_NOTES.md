# Novel_GRAND pipeline-fix overlay

This overlay fixes the current blockers that prevent a trustworthy end-to-end
hybrid LDPC + TAGS-GRAND run from showing up in the repo outputs.

## Fixed issues

1. **PyTorch checkpoint loading**
   - `novel_grand/models/training.py` now loads legacy checkpoints safely under
     PyTorch 2.6+ / 2.9.x by trying `weights_only=True` first and then falling
     back to `weights_only=False` for trusted local artifacts.
   - Newly saved bundles store normalization statistics as plain Python lists so
     future loads work with `weights_only=True`.

2. **Pipeline config propagation**
   - The collect / train / eval / report Slurm stages now honor
     `NOVEL_GRAND_CONFIG` instead of hardcoding `configs/fir_default.yaml`.
   - `slurm/submit_tags_grand_pipeline.sh` writes `outputs/_last_pipeline_run.json`
     so downstream status checks know which experiment directory is expected.

3. **Result-integrity visibility**
   - `tools/check_pipeline_status.sh` now checks the actual run selected by the
     pipeline and warns when train-shard SNRs do not match the selected config.

4. **Repo cleanup**
   - `tools/clean_tags_repo.sh` removes stale generated outputs, stale nested
     `outputs/.../outputs/...` trees, old Slurm logs, editable metadata, and
     overlay archives while preserving `.venv-fir` and probe logs.
