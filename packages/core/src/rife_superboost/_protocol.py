# SPDX-License-Identifier: MIT
"""Private backend ABI independent of VS, CUDA, and TensorRT Python types."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FrameView:
    """Borrowed planar RGB memory, kept alive by the calling frame request.

    Args:
        planes: R/G/B plane addresses in CPU memory, contiguous pixels within a row.
        strides: Positive row strides in bytes, including row padding.
        width: Active pixel width.
        height: Active pixel height.
        sample_bytes: Two for IEEE half, four for IEEE single precision.
    """

    planes: tuple[int, int, int]
    strides: tuple[int, int, int]
    width: int
    height: int
    sample_bytes: int


class Session(Protocol):
    """Per-node execution state owned until its last in-flight request ends."""

    def run(self, left: FrameView, right: FrameView, output: FrameView, timestep: float,
            *, frame_ids: tuple[int, int] | None = None, enabled: bool = True) -> None:
        """Args:
            left: Borrowed first RGB frame.
            right: Borrowed second RGB frame with matching format and dimensions.
            output: Writable borrowed RGB frame.
            timestep: Scalar interpolation time between the two source frames.
            frame_ids: Optional immutable frame identities scoped to this node.
            enabled: False repeats the left frame without inference or cache reset.
        """
        ...

    def close(self) -> None:
        """Release native state after all requests finish; safe to call repeatedly."""
        ...
