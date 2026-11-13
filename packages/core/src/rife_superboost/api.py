# SPDX-License-Identifier: MIT
"""VapourSynth graph construction; GPU libraries load only for a selected backend."""

from __future__ import annotations

import ctypes
from fractions import Fraction
from importlib.metadata import entry_points
from typing import Any, TYPE_CHECKING

from ._protocol import FrameView
from .errors import BackendNotInstalledError, DependencyError, RifeError
from .models import Model, resolve_model

if TYPE_CHECKING:
    import vapoursynth as vs


def capabilities() -> dict[str, Any]:
    """Report installed providers without loading GPU DLLs or creating a VS core."""
    return {"api_version": 2, "interface": "vapoursynth.VideoNode", "formats": ["RGBH", "RGBS"],
            "factor": {"min": 1, "max": 16, "integer_only": True}, "skip_prop": True,
            "precision": "configured by the backend, independent of VS storage format",
            "color_conversion": False, "backends": sorted(ep.name for ep in entry_points(group="rife_superboost.backends")),
            "models": sorted(ep.name for ep in entry_points(group="rife_superboost.models"))}


def _frame_view(frame: Any, writable: bool = False) -> FrameView:
    """Borrow a VS frame's plane addresses while its request holds strong references.

    Args:
        frame: VideoFrame kept alive by the caller.
        writable: Request writable output addresses rather than read addresses.
    """
    address = frame.get_write_ptr if writable else frame.get_read_ptr
    planes = tuple(ctypes.cast(address(c), ctypes.c_void_p).value for c in range(3))
    return FrameView(planes, tuple(frame.get_stride(c) for c in range(3)), frame.width, frame.height,
                     frame.format.bytes_per_sample)


def interpolate(clip: vs.VideoNode, *, model: str | Model = "4.25", backend: Any = "tensorrt",
                factor: int = 2, skip_prop: str | None = None) -> vs.VideoNode:
    """Create an integer-factor CFR node using prepared precision assets.

    Args:
        clip: Constant-format VapourSynth RGBH or RGBS VideoNode with known CFR.
        model: Installed model ID or an explicit Model.from_file(...) descriptor.
        backend: Installed backend name or its configuration instance.
        factor: Integer multiplier in 1..16. One returns the source unchanged.
        skip_prop: Optional left-frame integer property. Nonzero repeats the left
            frame for that pair; absent means zero. None always interpolates.

    The node owns its worker through the registered callback. Original source
    frames keep their positions; the last frame occupies all final phases.
    Output count and fps are multiplied by factor, preserving CFR duration.
    Skip decisions are read at request time, never during graph construction.
    Single-frame clips and factor=1 do not load model or GPU dependencies.
    """
    try:
        import vapoursynth as vs
    except ImportError as error:
        raise DependencyError("RIFE-SuperBoost requires an existing VapourSynth Python environment") from error
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("clip must be a vapoursynth.VideoNode")
    if type(factor) is not int or not 1 <= factor <= 16:
        raise RifeError("factor must be an integer in 1..16")
    if skip_prop is not None and (not isinstance(skip_prop, str) or not skip_prop or "\0" in skip_prop):
        raise RifeError("skip_prop must be None or a nonempty property name without NUL")
    if clip.format is None or clip.format.id not in (vs.RGBH, vs.RGBS):
        if clip.format is not None and clip.format.color_family == vs.YUV:
            raise NotImplementedError("Built-in YUV/RGB conversion is not implemented; supply VS RGBH/RGBS")
        raise RifeError("A constant RGBH or RGBS clip is required")
    if not 1 <= clip.width <= 8192 or not 1 <= clip.height <= 8192:
        raise RifeError("Source width and height must be 1..8192")
    if not 1 <= clip.num_frames <= 2147483647 // factor or clip.fps_num <= 0 or clip.fps_den <= 0:
        raise RifeError("A finite CFR clip with a supported frame count is required")
    rate = Fraction(clip.fps_num * factor, clip.fps_den)
    if rate.numerator > 2**63 - 1 or rate.denominator > 2**63 - 1:
        raise RifeError("Output CFR rate exceeds VapourSynth's signed 64-bit rational range")
    if factor == 1:
        return clip
    session = None
    if clip.num_frames > 1:
        identity = resolve_model(model)
        if isinstance(backend, str):
            matches = list(entry_points(group="rife_superboost.backends", name=backend))
            if len(matches) != 1:
                raise BackendNotInstalledError(f"Backend {backend!r} is not uniquely installed. Install RIFE-SuperBoost[tensorrt] for TensorRT.")
            backend = matches[0].load()()
        if not callable(getattr(backend, "create_session", None)):
            raise TypeError("backend must name a provider or implement create_session(model, width, height)")
        select_input = getattr(backend, "for_input", None)
        if callable(select_input):
            backend = select_input(clip.format.bytes_per_sample)
    core = vs.core
    following = clip[1:] + clip[-1] if clip.num_frames > 1 else clip
    timeline = core.std.Interleave([clip] * factor)
    next_timeline = core.std.Interleave([following] * factor)
    duration = 1 / rate
    if clip.num_frames > 1:
        session = backend.create_session(identity, clip.width, clip.height)

    def generate(n: int, f: list[Any]) -> Any:
        """Produce one frame from per-request VS inputs and node-owned state.

        Args:
            n: Output frame index; multiples of factor preserve source frames.
            f: Current/next source frames supplied by the upstream VS graph.
        """
        source, phase = divmod(n, factor)
        result = f[0].copy()
        if phase and source + 1 < clip.num_frames:
            enabled = True
            if skip_prop is not None:
                value = f[0].props.get(skip_prop, 0)
                if not isinstance(value, int):
                    raise RifeError(f"skip_prop {skip_prop!r} must contain one scalar integer, got {type(value).__name__}")
                enabled = value == 0
            session.run(_frame_view(f[0]), _frame_view(f[1]), _frame_view(result, True), phase / factor,
                        frame_ids=(source, source + 1), enabled=enabled)
        result.props["_DurationNum"] = duration.numerator
        result.props["_DurationDen"] = duration.denominator
        return result

    try:
        node = core.std.ModifyFrame(timeline, [timeline, next_timeline], generate)
        return core.std.AssumeFPS(node, fpsnum=rate.numerator, fpsden=rate.denominator)
    except Exception:
        if session is not None:
            session.close()
        raise
