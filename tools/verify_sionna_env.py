#!/usr/bin/env python3
import importlib
import json
import os
import platform
import sys
import time

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("cwd:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))

import torch
import sionna

print("torch_version:", torch.__version__)
print("sionna_version:", getattr(sionna, "__version__", "unknown"))
print("torch_cuda_is_available:", torch.cuda.is_available())
print("torch_num_threads:", torch.get_num_threads())
print("torch_num_interop_threads:", torch.get_num_interop_threads())

# small torch smoke test
x = torch.randn(256, 256)
y = x @ x.T
print("torch_smoke_shape:", tuple(y.shape))
print("torch_smoke_abs_mean:", float(y.abs().mean()))

modules = [
    "sionna.phy.fec.ldpc.encoding",
    "sionna.phy.fec.ldpc.decoding",
    "sionna.phy.channel.tr38901.tdl",
    "sionna.phy.channel.tr38901.cdl",
    "sionna.phy.channel.tr38901.umi",
    "sionna.phy.channel.tr38901.uma",
    "sionna.phy.nr.tb_encoder",
    "sionna.phy.nr.tb_decoder",
    "sionna.phy.nr.pusch_transmitter",
    "sionna.phy.nr.pusch_receiver",
]

timings = {}
for m in modules:
    t0 = time.time()
    importlib.import_module(m)
    timings[m] = round(time.time() - t0, 3)

print("import_timings_sec:", json.dumps(timings, indent=2, sort_keys=True))
print("STATUS: OK")
