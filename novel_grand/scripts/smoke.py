from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from novel_grand.config import load_config, run_root
from novel_grand.ldpc.effective_code import prepare_effective_pcm_cache
from novel_grand.sim.channel import NRSlotQAMLink
from novel_grand.ldpc.bp_trace import BPTraceRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = run_root(cfg)
    smoke_dir = out_root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepare_effective_pcm_cache(
            cfg,
            encoder_factory=lambda: NRSlotQAMLink(cfg).encoder,
            verbose=True,
        )
        link = NRSlotQAMLink(cfg)
        tracer = BPTraceRunner(link.encoder, cfg)

        ebn0_db = float(cfg["simulation"]["ebn0_db_list"][0])
        sample = link.sample(ebn0_db)
        trace = tracer.decode_with_trace(sample.llr_ch, sample.codeword_bits, sample.info_bits)
        tx_hard = sample.codeword_bits.reshape(-1).detach().cpu().numpy().astype("uint8")
        tx_syndrome_weight = tracer.graph_exact.syndrome_weight(tracer.graph_exact.syndrome_mask(tx_hard))

        summary = {
            "status": "ok",
            "k": link.k,
            "n": link.n,
            "bits_per_symbol": link.bits_per_symbol,
            "channel_mode": link.channel_mode,
            "ebn0_db": ebn0_db,
            "legacy_success": trace.legacy_success,
            "stop_iteration": trace.stop_iteration,
            "final_syndrome_weight": trace.snapshots[-1].syndrome_weight,
            "tx_codeword_exact_syndrome_weight": int(tx_syndrome_weight),
        }
        (smoke_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover
        summary = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (smoke_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
