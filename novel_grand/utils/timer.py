from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    seconds: float


@contextmanager
def timer():
    t0 = time.perf_counter()
    holder = {"seconds": 0.0}
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - t0
