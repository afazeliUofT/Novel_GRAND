from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from novel_grand.config import load_config, run_root
from novel_grand.ldpc.effective_code import prepare_effective_pcm_cache
from novel_grand.ldpc.bp_trace import BPTraceRunner
from novel_grand.sim.channel import NRSlotQAMLink
from novel_grand.utils.seed import set_global_seed



def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")



def _probe_worker(cfg: Dict, worker_id: int, ebn0_db: float) -> Dict:
    probe_cfg = cfg.get("probe", {})
    n_frames = int(probe_cfg.get("frames_per_worker_per_snr", 64))
    seed = int(cfg["system"]["seed"]) + 9000 * worker_id + int(round(ebn0_db * 100))

    set_global_seed(seed)
    torch.set_num_threads(int(cfg["system"]["torch_threads_per_worker"]))

    link = NRSlotQAMLink(cfg)
    tracer = BPTraceRunner(link.encoder, cfg)

    out_root = run_root(cfg)
    shard_dir = _ensure_dir(out_root / "probe" / "shards")

    rows: List[Dict] = []
    for frame_idx in range(n_frames):
        t0 = time.perf_counter()
        sample = link.sample(ebn0_db)
        trace = tracer.decode_with_trace(sample.llr_ch, sample.codeword_bits, sample.info_bits)
        runtime_ms = 1000.0 * (time.perf_counter() - t0)
        final = trace.snapshots[-1]
        rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "worker_id": int(worker_id),
                "frame_idx": int(frame_idx),
                "legacy_success": int(trace.legacy_success),
                "legacy_detected_failure": int(not trace.legacy_success),
                "success_exact": int(trace.legacy_success and (final.hard == trace.true_codeword).all()),
                "valid_codeword": int(final.syndrome_mask == 0),
                "undetected_error": int(final.syndrome_mask == 0 and not (final.hard == trace.true_codeword).all()),
                "stop_iteration": int(trace.stop_iteration),
                "final_syndrome_weight": int(final.syndrome_weight),
                "runtime_ms": float(runtime_ms),
            }
        )

    shard_path = shard_dir / f"legacy_probe_worker{worker_id:02d}_snr{ebn0_db:.2f}.jsonl"
    _write_jsonl(shard_path, rows)
    return {
        "worker_id": int(worker_id),
        "ebn0_db": float(ebn0_db),
        "n_frames": int(n_frames),
        "shard_path": str(shard_path),
    }



def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for ebn0_db, g in df.groupby("ebn0_db"):
        fail_mask = g["legacy_detected_failure"] > 0
        fail_weights = g.loc[fail_mask, "final_syndrome_weight"]
        out_rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "n_frames": int(len(g)),
                "detected_failure_rate": float(g["legacy_detected_failure"].mean()),
                "exact_success_rate": float(g["success_exact"].mean()),
                "valid_codeword_rate": float(g["valid_codeword"].mean()),
                "undetected_error_rate": float(g["undetected_error"].mean()),
                "avg_stop_iteration": float(g["stop_iteration"].mean()),
                "avg_runtime_ms": float(g["runtime_ms"].mean()),
                "avg_final_syndrome_weight_on_fail": float(fail_weights.mean()) if len(fail_weights) else 0.0,
            }
        )
    return pd.DataFrame(out_rows).sort_values("ebn0_db").reset_index(drop=True)



def _recommend_ebn0(summary: pd.DataFrame, cfg: Dict) -> List[float]:
    probe_cfg = cfg.get("probe", {})
    target_min = float(probe_cfg.get("target_failure_min", 0.01))
    target_max = float(probe_cfg.get("target_failure_max", 0.20))
    recommend_count = int(probe_cfg.get("recommend_count", 5))
    target_mid = 0.5 * (target_min + target_max)

    cand = summary[
        (summary["detected_failure_rate"] >= target_min)
        & (summary["detected_failure_rate"] <= target_max)
    ].copy()
    if not cand.empty:
        return [float(x) for x in cand["ebn0_db"].tolist()[:recommend_count]]

    fallback = summary.copy()
    fallback["distance_to_target" ] = (fallback["detected_failure_rate"] - target_mid).abs()
    fallback = fallback.sort_values(["distance_to_target", "ebn0_db"]).head(recommend_count)
    return sorted(float(x) for x in fallback["ebn0_db"].tolist())



def _write_recommended_config(cfg: Dict, recommended_ebn0: List[float], out_path: Path) -> None:
    cfg2 = copy.deepcopy(cfg)
    cfg2["experiment_name"] = "fir_tags_grand_autotuned"
    cfg2.setdefault("simulation", {})["ebn0_db_list"] = [float(x) for x in recommended_ebn0]
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False)



def _plot_probe(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    plt.plot(summary["ebn0_db"], summary["detected_failure_rate"], marker="o")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Legacy detected failure rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "legacy_detected_failure_rate_vs_ebn0.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(summary["ebn0_db"], summary["undetected_error_rate"], marker="o")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Legacy undetected error rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "legacy_undetected_error_rate_vs_ebn0.png", dpi=180)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    report_dir = _ensure_dir(out_root / "reports")

    prepare_effective_pcm_cache(
        cfg,
        encoder_factory=lambda: NRSlotQAMLink(cfg).encoder,
        verbose=True,
    )

    jobs = []
    for ebn0_db in cfg["simulation"]["ebn0_db_list"]:
        for worker_id in range(int(cfg["system"]["num_workers"])):
            jobs.append((cfg, worker_id, float(ebn0_db)))

    manifests = []
    with ProcessPoolExecutor(max_workers=int(cfg["system"]["num_workers"])) as ex:
        futs = [ex.submit(_probe_worker, *job) for job in jobs]
        for fut in as_completed(futs):
            manifests.append(fut.result())

    manifest_path = out_root / "probe" / "probe_manifest.jsonl"
    _write_jsonl(manifest_path, manifests)

    rows: List[Dict] = []
    for item in manifests:
        with Path(item["shard_path"]).open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    summary = _summarize(df)
    summary.to_csv(report_dir / "legacy_probe_summary.csv", index=False)
    df.to_csv(report_dir / "legacy_probe_rows_all.csv", index=False)
    _plot_probe(summary, report_dir)

    recommended_ebn0 = _recommend_ebn0(summary, cfg)
    recommended = {
        "recommended_ebn0_db_list": recommended_ebn0,
        "selection_rule": {
            "target_failure_min": float(cfg.get("probe", {}).get("target_failure_min", 0.01)),
            "target_failure_max": float(cfg.get("probe", {}).get("target_failure_max", 0.20)),
        },
    }
    with (report_dir / "recommended_ebn0.json").open("w", encoding="utf-8") as f:
        json.dump(recommended, f, indent=2)

    _write_recommended_config(cfg, recommended_ebn0, report_dir / "recommended_full_config.yaml")

    print(summary.to_string(index=False))
    print()
    print(json.dumps({
        "status": "ok",
        "n_jobs": len(manifests),
        "out_root": str(out_root),
        "recommended_ebn0_db_list": recommended_ebn0,
    }, indent=2))


if __name__ == "__main__":
    main()
