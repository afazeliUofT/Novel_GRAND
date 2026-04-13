# Novel GRAND package for FIR

This overlay adds a **Sionna-based 5G NR LDPC + TAGS-GRAND-Lite** research pipeline to your FIR repo and existing `.venv-fir`.

## Mechanism in one paragraph

The package keeps a **legacy 5G NR LDPC decoder** as the main decoder. For the rare frames that still have a **non-zero syndrome after the LDPC iteration cap**, it stores the full BP trajectory, learns which iteration snapshot is the best rescue point, learns which bits are most likely still wrong, and then runs an **exact syndrome-checked GRAND-style search** over a mix of **single-bit** and **group** flip primitives induced by the Tanner graph and OFDM/QAM structure.

## Important implementation detail

For Sionna 5G NR LDPC, `LDPC5GEncoder.pcm` is the **mother-code** parity-check matrix before rate-matching, while the encoder output has the **rate-matched** length `n`. Because of that, this package derives and caches an **effective parity-check matrix for the transmitted rate-matched code** the first time it runs. That cache is written to:

```text
outputs/_effective_code_cache/
```

The smoke script prepares this cache automatically before running the trace code.

## Install into the repo

From:

```bash
cd /home/rsadve1/scratch/Novel_GRAND
```

unzip the overlay at repo root, then install the package into the already-working venv:

```bash
source env/activate_fir.sh
python -m pip install -e .
```

## Safe first check

Run the interactive smoke wrapper:

```bash
bash tools/check_tags_package.sh
```

This wrapper always leaves a log in:

```text
probe_outputs/check_tags_package_<timestamp>.log
```

## Slurm usage

Smoke:

```bash
sbatch slurm/00_smoke_tags_grand.sbatch
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
smoke/             smoke summary
train/shards/      raw training shards from failed LDPC traces
models/            trained snapshot selector + bit ranker
eval/shards/       per-frame evaluation rows
eval/sampled_failures/ representative failure traces
reports/           merged CSVs, plots, markdown summary
```


## v6 note
The default evaluation now includes cap-matched non-AI baselines so that any TAGS gain can be judged against equal query budgets.
