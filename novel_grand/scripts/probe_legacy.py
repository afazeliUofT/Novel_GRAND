from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from novel_grand.config import load_config, run_root
from novel_grand.ldpc.bp_trace import BPTraceRunner
from novel_grand.ldpc.effective_code import prepare_effective_pcm_cache
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

    link = NRSlotQAMLink(cfg, seed=seed)
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


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
        / denom
    )
    return float(center - half), float(center + half)


def _pava_nonincreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    blocks = [
        {"start": i, "end": i, "weight": float(w[i]), "value": float(y[i])}
        for i in range(len(y))
    ]

    i = 0
    while i < len(blocks) - 1:
        if blocks[i]["value"] < blocks[i + 1]["value"] - 1e-15:
            w_new = blocks[i]["weight"] + blocks[i + 1]["weight"]
            v_new = (
                blocks[i]["weight"] * blocks[i]["value"]
                + blocks[i + 1]["weight"] * blocks[i + 1]["value"]
            ) / w_new
            blocks[i : i + 2] = [
                {
                    "start": blocks[i]["start"],
                    "end": blocks[i + 1]["end"],
                    "weight": w_new,
                    "value": v_new,
                }
            ]
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.empty_like(y, dtype=float)
    for b in blocks:
        out[b["start"] : b["end"] + 1] = b["value"]
    return out


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for ebn0_db, g in df.groupby("ebn0_db", sort=True):
        n_frames = int(len(g))
        fail_count = int(g["legacy_detected_failure"].sum())
        undetected_count = int(g["undetected_error"].sum())

        fail_ci_low, fail_ci_high = _wilson_interval(fail_count, n_frames)
        und_ci_low, und_ci_high = _wilson_interval(undetected_count, n_frames)

        fail_mask = g["legacy_detected_failure"] > 0
        fail_weights = g.loc[fail_mask, "final_syndrome_weight"]

        out_rows.append(
            {
                "ebn0_db": float(ebn0_db),
                "n_frames": n_frames,
                "detected_failure_rate": float(g["legacy_detected_failure"].mean()),
                "detected_failure_rate_ci_low": fail_ci_low,
                "detected_failure_rate_ci_high": fail_ci_high,
                "exact_success_rate": float(g["success_exact"].mean()),
                "valid_codeword_rate": float(g["valid_codeword"].mean()),
                "undetected_error_rate": float(g["undetected_error"].mean()),
                "undetected_error_rate_ci_low": und_ci_low,
                "undetected_error_rate_ci_high": und_ci_high,
                "avg_stop_iteration": float(g["stop_iteration"].mean()),
                "avg_runtime_ms": float(g["runtime_ms"].mean()),
                "avg_final_syndrome_weight_on_fail": float(fail_weights.mean()) if len(fail_weights) else 0.0,
            }
        )

    summary = pd.DataFrame(out_rows).sort_values("ebn0_db").reset_index(drop=True)
    summary["detected_failure_rate_smoothed"] = _pava_nonincreasing(
        summary["detected_failure_rate"].to_numpy(dtype=float),
        summary["n_frames"].to_numpy(dtype=float),
    )
    return summary


def _recommend_ebn0(summary: pd.DataFrame, cfg: Dict) -> List[float]:
    probe_cfg = cfg.get("probe", {})
    target_min = float(probe_cfg.get("target_failure_min", 0.01))
    target_max = float(probe_cfg.get("target_failure_max", 0.20))
    recommend_count = int(probe_cfg.get("recommend_count", 5))
    target_mid = 0.5 * (target_min + target_max)

    cand = summary[
        (summary["detected_failure_rate_smoothed"] >= target_min)
        & (summary["detected_failure_rate_smoothed"] <= target_max)
    ].copy()

    if cand.empty:
        cand = summary.copy()

    cand["distance_to_target"] = (cand["detected_failure_rate_smoothed"] - target_mid).abs()
    cand = cand.sort_values(["distance_to_target", "ebn0_db"]).head(recommend_count)
    return sorted(float(x) for x in cand["ebn0_db"].tolist())


def _write_recommended_config(cfg: Dict, recommended_ebn0: List[float], out_path: Path) -> None:
    probe_cfg = cfg.get("probe", {})
    base_cfg_path = probe_cfg.get("base_full_config")
    if base_cfg_path:
        try:
            cfg2 = load_config(base_cfg_path)
        except Exception:
            cfg2 = copy.deepcopy(cfg)
    else:
        cfg2 = copy.deepcopy(cfg)
    cfg2["experiment_name"] = "fir_tags_grand_autotuned"
    cfg2.setdefault("simulation", {})["ebn0_db_list"] = [float(x) for x in recommended_ebn0]
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False)


def _plot_probe(summary: pd.DataFrame, out_dir: Path, cfg: Dict) -> None:
    target_min = float(cfg.get("probe", {}).get("target_failure_min", 0.01))
    target_max = float(cfg.get("probe", {}).get("target_failure_max", 0.20))

    plt.figure()
    plt.fill_between(
        summary["ebn0_db"],
        target_min,
        target_max,
        alpha=0.15,
        label="target rescue band",
    )
    y = summary["detected_failure_rate"]
    yerr_low = y - summary["detected_failure_rate_ci_low"]
    yerr_high = summary["detected_failure_rate_ci_high"] - y
    plt.errorbar(
        summary["ebn0_db"],
        y,
        yerr=[yerr_low, yerr_high],
        fmt="o-",
        capsize=3,
        label="raw failure rate",
    )
    plt.plot(
        summary["ebn0_db"],
        summary["detected_failure_rate_smoothed"],
        marker="s",
        label="monotone-smoothed failure rate",
    )
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Legacy detected failure rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "legacy_detected_failure_rate_vs_ebn0.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(summary["ebn0_db"], summary["exact_success_rate"], marker="o")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Legacy exact success rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "legacy_exact_success_rate_vs_ebn0.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(summary["ebn0_db"], summary["avg_stop_iteration"], marker="o")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Average stop iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "legacy_avg_stop_iteration_vs_ebn0.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(summary["ebn0_db"], summary["avg_runtime_ms"], marker="o")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Average runtime per frame [ms]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "legacy_avg_runtime_ms_vs_ebn0.png", dpi=180)
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
    _plot_probe(summary, report_dir, cfg)

    recommended_ebn0 = _recommend_ebn0(summary, cfg)
    recommended = {
        "recommended_ebn0_db_list": recommended_ebn0,
        "selection_rule": {
            "target_failure_min": float(cfg.get("probe", {}).get("target_failure_min", 0.01)),
            "target_failure_max": float(cfg.get("probe", {}).get("target_failure_max", 0.20)),
            "used_smoothed_failure_curve": True,
        },
    }

    with (report_dir / "recommended_ebn0.json").open("w", encoding="utf-8") as f:
        json.dump(recommended, f, indent=2)

    _write_recommended_config(
        cfg,
        recommended_ebn0,
        report_dir / "recommended_full_config.yaml",
    )

    print(summary.to_string(index=False))
    print()
    print(
        json.dumps(
            {
                "status": "ok",
                "n_jobs": len(manifests),
                "out_root": str(out_root),
                "recommended_ebn0_db_list": recommended_ebn0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
