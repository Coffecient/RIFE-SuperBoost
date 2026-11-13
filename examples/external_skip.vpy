# SPDX-License-Identifier: MIT
"""Use an external per-frame decision without relying on a persistent Python flag."""

import vapoursynth as vs
from rife_superboost import interpolate
from rife_superboost_tensorrt import TensorRT

# Replace this source with an existing RGBH/RGBS node. The attribute is evaluated
# when each frame is requested, even after the graph-construction scope returns.
rgb = vs.core.std.BlankClip(width=1280, height=720, length=48, format=vs.RGBH, fpsnum=24, fpsden=1)


def decision(n, f):
    """Args:
        n: Left source frame index for the external decision.
        f: Source frame whose properties are copied for this request.
    """
    result = f.copy()
    result.props["ExternalDisableRIFE"] = int(20 <= n < 24)  # Example external decision.
    return result


flagged = vs.core.std.ModifyFrame(rgb, rgb, decision)
model = "4.25"  # Select an independently installed model wheel.
backend = TensorRT(precision="fp16")
output = interpolate(flagged, model=model, backend=backend, factor=3, skip_prop="ExternalDisableRIFE")
output.set_output()
