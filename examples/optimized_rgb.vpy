# SPDX-License-Identifier: MIT
"""Installed-model example; replace the synthetic source before use."""

import vapoursynth as vs
from rife_superboost import interpolate
from rife_superboost_tensorrt import TensorRT

# Supply an existing constant RGBH/RGBS clip here; color conversion stays external.
rgb = vs.core.std.BlankClip(width=1280, height=720, length=48, format=vs.RGBH, fpsnum=24, fpsden=1)
model = "4.25"  # Select an independently installed model wheel.
backend = TensorRT(
    # NVIDIA libraries and a writable per-user engine cache are located automatically.
    encoding_cache=True,  # Consume pre-exported encoder/synthesis assets.
    frame_cache_mib=256,  # RGB/feature slots only; excludes engines/workspace/staging.
    cache_slots=4,  # Additional hard slot cap; at least two must fit the byte budget.
    cuda_graph=False,  # Optional fixed-address synthesis capture.
    precision="fp16",  # Half convolutions, protected FP32 accumulation/geometry.
)
interpolate(rgb, model=model, backend=backend, factor=2, skip_prop=None).set_output()
