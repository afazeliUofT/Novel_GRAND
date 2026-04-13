# Novel GRAND package for FIR

This is a **full standalone package** for a Sionna-based **5G NR LDPC + MaskDiff-GRAND** study on FIR.

## Mechanism in one paragraph

The package keeps the **legacy 5G NR LDPC decoder** as the primary decoder. Only frames with **non-zero syndrome after the LDPC iteration cap** go to the rescue path. The rescue path first runs a strong **final-LLR GRAND guard** on the final LDPC snapshot. If that fails, it uses a **learned snapshot selector** to choose a promising failed-BP snapshot and then runs a **verifier-guided masked-diffusion set generator** on a conservative shortlist of risky bits. The diffusion stage proposes full correction sets rather than one-step flips, and every candidate is still validated by **exact syndrome checking** on the transmitted rate-matched code. A conservative **best-syndrome LLR fallback** remains in place after the AI stage.

## Why this package exists

The previous FlowSearch-GRAND path showed that the selector was useful, but the **AI rescue stage itself still contributed almost no exact rescues**. This package replaces that path with a different AI architecture inspired by recent discrete diffusion and inference-time search advances:

- a **teacher-aligned snapshot selector**,
- a **masked discrete denoiser** over a shortlist of candidate bits,
- a **parallel unmasking sampler** that proposes whole correction sets,
- a lightweight **consensus reconditioning** round,
- and a **local exact-repair** step under the same fixed rescue budget.

This is meant to test whether **whole-set generative search** is more effective than local tree expansion over individual actions.

## Important implementation detail

For Sionna 5G NR LDPC, `LDPC5GEncoder.pcm` is the **mother-code** parity-check matrix before rate-matching, while the encoder output has the **rate-matched** transmitted length `n`. Because of that, this package derives and caches an **effective parity-check matrix for the transmitted rate-matched code** on first use.

That cache is written to:

```text
outputs/_effective_code_cache/
```

## Install into the repo

From:

```bash
cd /home/rsadve1/scratch/Novel_GRAND
```

unzip the package at repo root, then install it into the already-working venv:

```bash
source env/activate_fir.sh
python -m pip install -e .
```

## Safe first check

Run the interactive smoke wrapper:

```bash
bash tools/check_tags_package.sh
```

This leaves a log in:

```text
probe_outputs/check_tags_package_<timestamp>.log
```

## Slurm usage

Smoke:

```bash
sbatch slurm/00_smoke_tags_grand.sbatch
```

Legacy probe:

```bash
bash slurm/submit_legacy_probe.sh
```

Full pipeline:

```bash
bash slurm/submit_tags_grand_pipeline.sh
```

## Output layout

All outputs land under:

```text
outputs/<experiment_name>/
```

Key folders:

```text
smoke/                 smoke summary
probe/shards/          legacy-LDPC probe shards
train/shards/          raw training shards from failed LDPC traces
models/                trained snapshot selector + masked-diffusion model
eval/shards/           per-frame evaluation rows
eval/sampled_failures/ representative failure traces
reports/               merged CSVs, plots, markdown summary
```
