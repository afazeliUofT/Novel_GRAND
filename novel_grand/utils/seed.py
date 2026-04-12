from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Sionna PHY maintains its own global seed. If left untouched, workers can
    # silently generate identical Monte Carlo streams even when torch/numpy are
    # seeded differently. We set it whenever Sionna is importable.
    try:  # pragma: no cover - only exercised on FIR with Sionna installed
        import sionna
        if hasattr(sionna, 'phy') and hasattr(sionna.phy, 'config'):
            sionna.phy.config.seed = seed
        elif hasattr(sionna, 'config'):
            sionna.config.seed = seed
    except Exception:
        pass
