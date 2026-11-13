# SPDX-License-Identifier: MIT
"""Reserved VapourSynth color-conversion boundary."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import vapoursynth as vs


def convert_color(clip: vs.VideoNode, *, format: int) -> vs.VideoNode:
    """Reserve YUV/RGB input/output conversion without an implicit fallback.

    Args:
        clip: Existing VapourSynth VideoNode to convert.
        format: Requested VapourSynth format ID, including a YUV output format.

    This entry point always raises, without loading GPU libraries or inspecting
    frames. Use external VS/zimg conversion until this feature is implemented.
    """
    raise NotImplementedError("Built-in YUV/RGB color conversion is not implemented; convert externally with VapourSynth/zimg")
