# SPDX-License-Identifier: MIT
"""M2 reference example; replace the paths and synthetic source before use."""

import vapoursynth as vs
from rife_superboost import Model, interpolate
from rife_superboost_tensorrt import TensorRT

# The source must already be RGBH/RGBS; color conversion is outside this example.
rgb = vs.core.std.BlankClip(width=1280, height=720, length=48, format=vs.RGBH, fpsnum=24, fpsden=1)
model = Model.from_file(r"C:\Models\rife-4.25-reference-fp32.model.json")
backend = TensorRT(
    runtime_dir=r"C:\NVIDIA\TensorRT-10.16.1.11\bin",  # Explicit NVIDIA runtime directory.
    cache_dir=r"C:\VideoCache\RIFE-SuperBoost",  # Writable engine cache outside package data.
    device=0,  # CUDA device ordinal.
    workspace_mib=512,  # Native engine-builder workspace budget.
    precision="fp32",  # Historical reference graphs contain no FP16 variant.
)
interpolate(rgb, model=model, backend=backend).set_output()
