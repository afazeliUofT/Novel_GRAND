from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from novel_grand.config import load_config, run_root
from novel_grand.ldpc.effective_code import prepare_effective_pcm_cache
from novel_grand.sim.channel import NRSlotQAMLink
from novel_grand.sim.worker import evaluate_worker
from novel_grand.utils.io import write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    prepare_effective_pcm_cache(
        cfg,
        encoder_factory=lambda: NRSlotQAMLink(cfg).encoder,
        verbose=True,
    )

    jobs = []
    for ebn0_db in cfg["simulation"]["ebn0_db_list"]:
        for worker_id in range(int(cfg["system"]["num_workers"])):
            jobs.append((cfg, worker_id, float(ebn0_db)))

    results = []
    with ProcessPoolExecutor(max_workers=int(cfg["system"]["num_workers"])) as ex:
        futs = [ex.submit(evaluate_worker, *job) for job in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())

    write_jsonl(out_root / "eval" / "evaluation_manifest.jsonl", results)
    print(json.dumps({"status": "ok", "n_jobs": len(results), "out_root": str(out_root)}, indent=2))


if __name__ == "__main__":
    main()
