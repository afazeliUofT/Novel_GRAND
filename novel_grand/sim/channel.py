from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class LinkSample:
    info_bits: torch.Tensor
    codeword_bits: torch.Tensor
    llr_ch: torch.Tensor
    ebn0_db: float


class NRSlotQAMLink:
    def __init__(self, cfg: Dict, seed: Optional[int] = None):
        self.cfg = cfg
        self.device = cfg["system"]["device"]
        self.seed = None if seed is None else int(seed)

        try:
            from sionna.phy.fec.ldpc import LDPC5GEncoder
            from sionna.phy.mapping import Mapper, Demapper, BinarySource
            from sionna.phy.ofdm.resource_grid import ResourceGrid
            from sionna.phy.channel.ofdm_channel import OFDMChannel
            from sionna.phy.channel.tr38901.tdl import TDL
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Sionna imports failed. Activate the Novel_GRAND project venv first: source env/activate_fir.sh"
            ) from exc

        nr = cfg["nr"]
        ch = cfg["channel"]
        self.k = int(nr["k"])
        self.n = int(nr["n"])
        self.bits_per_symbol = int(nr["bits_per_symbol"])
        self.num_ofdm_symbols = int(nr["num_ofdm_symbols"])
        self.fft_size = int(nr["fft_size"])
        self.cyclic_prefix_length = int(nr["cyclic_prefix_length"])
        self.num_data_symbols = self.num_ofdm_symbols * self.fft_size
        expected_n = self.num_data_symbols * self.bits_per_symbol
        if expected_n != self.n:
            raise ValueError(
                f"Resource grid carries {expected_n} coded bits but config requests n={self.n}."
            )

        self.encoder = LDPC5GEncoder(
            k=self.k,
            n=self.n,
            num_bits_per_symbol=self.bits_per_symbol,
            device=self.device,
        )

        # Use an explicit BinarySource seed when supported. Falling back to a
        # local torch generator still guarantees worker-unique Monte Carlo bits.
        self._binary_source = None
        try:
            if self.seed is not None:
                self._binary_source = BinarySource(seed=self.seed, device=self.device)
            else:
                self._binary_source = BinarySource(device=self.device)
        except TypeError:
            self._binary_source = None

        self._torch_gen = torch.Generator(device='cpu')
        if self.seed is not None:
            self._torch_gen.manual_seed(self.seed)

        self.mapper = Mapper("qam", self.bits_per_symbol, device=self.device)
        self.demapper = Demapper("maxlog", "qam", self.bits_per_symbol, device=self.device)

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=float(nr["subcarrier_spacing_hz"]),
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=self.cyclic_prefix_length,
            num_guard_carriers=tuple(int(x) for x in nr["num_guard_carriers"]),
            dc_null=bool(nr["dc_null"]),
            pilot_pattern=None,
            device=self.device,
        )

        self.channel_mode = str(ch["mode"]).lower()
        self.eps_equalizer = float(ch.get("eps_equalizer", 1e-8))
        if self.channel_mode == "tdl":
            tdl_kwargs = dict(
                model=str(ch["delay_profile"]),
                delay_spread=float(ch["delay_spread_s"]),
                carrier_frequency=float(ch["carrier_frequency_hz"]),
                min_speed=float(ch["min_speed_mps"]),
                max_speed=float(ch["max_speed_mps"]),
                num_rx_ant=1,
                num_tx_ant=1,
                device=self.device,
            )
            if self.seed is not None:
                tdl_kwargs["seed"] = self.seed
            try:
                tdl = TDL(**tdl_kwargs)
            except TypeError:
                tdl_kwargs.pop("seed", None)
                tdl = TDL(**tdl_kwargs)
            self.channel = OFDMChannel(
                channel_model=tdl,
                resource_grid=self.resource_grid,
                normalize_channel=True,
                return_channel=True,
                device=self.device,
            )
        elif self.channel_mode == "awgn":
            self.channel = None
        else:
            raise ValueError(f"Unsupported channel mode: {self.channel_mode}")

    @property
    def coderate(self) -> float:
        return self.k / self.n

    def ebn0_to_no(self, ebn0_db: float) -> float:
        ebn0_lin = 10.0 ** (ebn0_db / 10.0)
        esn0_lin = ebn0_lin * self.coderate * self.bits_per_symbol
        return float(1.0 / esn0_lin)

    def _sample_info_bits(self) -> torch.Tensor:
        if self._binary_source is not None:
            return self._binary_source([1, self.k])
        # Fallback path if BinarySource does not expose a seed parameter in the
        # installed Sionna version.
        bits = torch.randint(
            low=0,
            high=2,
            size=(1, self.k),
            generator=self._torch_gen,
            dtype=torch.int64,
        ).to(self.device)
        return bits.to(torch.float32)

    def sample(self, ebn0_db: float) -> LinkSample:
        b = self._sample_info_bits()
        c = self.encoder(b)
        x = self.mapper(c).reshape(1, 1, 1, self.num_ofdm_symbols, self.fft_size)

        no = self.ebn0_to_no(ebn0_db)
        if self.channel_mode == "tdl":
            y, h = self.channel(x, no)
            h_siso = h[:, 0, 0, 0, 0, :, :]
            y_siso = y[:, 0, 0, :, :] / (h_siso + self.eps_equalizer)
            no_eff = no / (torch.abs(h_siso) ** 2 + self.eps_equalizer)
            y_flat = y_siso.reshape(1, -1)
            no_flat = no_eff.real.reshape(1, -1)
        else:
            noise = torch.sqrt(torch.tensor(no / 2.0, dtype=torch.float32, device=x.device))
            w_real = torch.randn(x.real.shape, generator=self._torch_gen, dtype=x.real.dtype).to(x.device)
            w_imag = torch.randn(x.real.shape, generator=self._torch_gen, dtype=x.real.dtype).to(x.device)
            w = noise * (w_real + 1j * w_imag)
            y_flat = (x + w)[:, 0, 0, :, :].reshape(1, -1)
            no_flat = torch.full_like(y_flat.real, fill_value=no)

        llr_ch = self.demapper(y_flat, no_flat)
        return LinkSample(
            info_bits=b.reshape(-1),
            codeword_bits=c.reshape(-1),
            llr_ch=llr_ch.reshape(-1),
            ebn0_db=float(ebn0_db),
        )
