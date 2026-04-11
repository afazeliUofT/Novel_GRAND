from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from novel_grand.ldpc.effective_code import load_or_build_effective_pcm, projected_transmitted_pcm
from novel_grand.ldpc.tanner import TannerGraph


@dataclass
class Snapshot:
    iter_idx: int
    posterior: np.ndarray
    hard: np.ndarray
    syndrome_mask: int
    syndrome_weight: int
    structure_syndrome_mask: int
    unsat_deg: np.ndarray
    change_fraction: float
    cumulative_flip_count: np.ndarray


@dataclass
class TraceResult:
    llr_ch: np.ndarray
    true_codeword: np.ndarray
    info_bits: np.ndarray
    snapshots: List[Snapshot]
    stop_iteration: int
    legacy_success: bool


class BPTraceRunner:
    def __init__(self, encoder, cfg):
        self.encoder = encoder
        self.cfg = cfg
        self.graph_exact = TannerGraph(load_or_build_effective_pcm(encoder, cfg))
        self.graph_struct = TannerGraph(projected_transmitted_pcm(encoder))
        self.graph = self.graph_exact

        try:
            from sionna.phy.fec.ldpc import LDPC5GDecoder
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not import Sionna LDPC5GDecoder. Activate the FIR venv first."
            ) from exc

        legacy_cfg = cfg["legacy_ldpc"]
        device = cfg["system"]["device"]
        self.decoder_soft = LDPC5GDecoder(
            encoder=encoder,
            cn_update=legacy_cfg["cn_update"],
            cn_schedule=legacy_cfg["cn_schedule"],
            hard_out=False,
            return_infobits=False,
            num_iter=1,
            llr_max=legacy_cfg["llr_max"],
            return_state=True,
            device=device,
        )
        self.decoder_info = LDPC5GDecoder(
            encoder=encoder,
            cn_update=legacy_cfg["cn_update"],
            cn_schedule=legacy_cfg["cn_schedule"],
            hard_out=True,
            return_infobits=True,
            num_iter=2,
            llr_max=max(legacy_cfg["llr_max"], 20.0),
            return_state=False,
            device=device,
        )
        self.max_iter = int(legacy_cfg["num_iter"])
        self.early_stop = bool(legacy_cfg["early_stop"])

    def decode_with_trace(
        self,
        llr_ch: torch.Tensor,
        true_codeword: torch.Tensor,
        info_bits: torch.Tensor,
    ) -> TraceResult:
        llr_ch = llr_ch.reshape(1, -1)
        true_codeword = true_codeword.reshape(-1).detach().cpu().numpy().astype(np.uint8)
        info_bits_np = info_bits.reshape(-1).detach().cpu().numpy().astype(np.uint8)
        llr_np = llr_ch.reshape(-1).detach().cpu().numpy().astype(np.float32)

        msg_v2c = None
        snapshots: List[Snapshot] = []
        prev_sign: Optional[np.ndarray] = None
        cumulative_flip_count = np.zeros(self.graph_exact.n, dtype=np.int16)

        for it in range(1, self.max_iter + 1):
            post, msg_v2c = self.decoder_soft(llr_ch, num_iter=1, msg_v2c=msg_v2c)
            post = post.reshape(-1).detach().cpu().numpy().astype(np.float32)
            hard = (post > 0).astype(np.uint8)
            synd = self.graph_exact.syndrome_mask(hard)
            struct_synd = self.graph_struct.syndrome_mask(hard)
            unsat_deg = self.graph_struct.unsatisfied_check_counts(struct_synd)

            if prev_sign is None:
                change_fraction = 0.0
            else:
                changed = (np.sign(post) != np.sign(prev_sign)).astype(np.int16)
                cumulative_flip_count = cumulative_flip_count + changed
                change_fraction = float(changed.mean())
            prev_sign = post.copy()

            snapshots.append(
                Snapshot(
                    iter_idx=it,
                    posterior=post,
                    hard=hard,
                    syndrome_mask=synd,
                    syndrome_weight=self.graph_exact.syndrome_weight(synd),
                    structure_syndrome_mask=struct_synd,
                    unsat_deg=unsat_deg,
                    change_fraction=change_fraction,
                    cumulative_flip_count=cumulative_flip_count.copy(),
                )
            )
            if synd == 0 and self.early_stop:
                break

        legacy_success = bool(snapshots[-1].syndrome_mask == 0)
        return TraceResult(
            llr_ch=llr_np,
            true_codeword=true_codeword,
            info_bits=info_bits_np,
            snapshots=snapshots,
            stop_iteration=int(snapshots[-1].iter_idx),
            legacy_success=legacy_success,
        )

    def decode_info_from_codeword(self, codeword_bits: np.ndarray) -> np.ndarray:
        llr = torch.tensor(
            (2.0 * codeword_bits.astype(np.float32) - 1.0) * 30.0,
            dtype=torch.float32,
            device=self.decoder_info.device,
        ).reshape(1, -1)
        with torch.no_grad():
            out = self.decoder_info(llr).reshape(-1).detach().cpu().numpy()
        return out.astype(np.uint8)
