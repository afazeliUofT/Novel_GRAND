# Novel GRAND package for FIR

This is a **full standalone package** for a Sionna-based **5G NR LDPC + FlowSearch-GRAND** study on FIR.

## Mechanism in one paragraph

The package keeps the **legacy 5G NR LDPC decoder** as the primary decoder. Only frames with **non-zero syndrome after the LDPC iteration cap** go to the rescue path. The rescue path first runs a strong **final-LLR GRAND guard** on the final LDPC snapshot. If that fails, it uses a **learned snapshot selector** to seed a **multi-snapshot policy-value tree search** over exact syndrome-preserving correction patterns. That search is followed by a conservative **best-syndrome LLR fallback**. Every rescue candidate is still validated by **exact syndrome checking** on the transmitted rate-matched code.

## Why this package exists

The previous TAGS-GRAND path showed that the overall pipeline was working, but the **AI stage itself was contributing almost no exact rescues**. This package replaces that path with a different AI architecture:

- a **teacher-aligned snapshot selector**,
- a **learned action prior** for the next flip action,
- a **learned state-value model** for partial correction patterns,
- a **best-first multi-snapshot tree search** under a fixed rescue budget.

This is meant to test whether **set-level search guidance** is more effective than the earlier static bit-ranking path.

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
models/                trained snapshot selector + action prior + state value model
eval/shards/           per-frame evaluation rows
eval/sampled_failures/ representative failure traces
reports/               merged CSVs, plots, markdown summary
```
