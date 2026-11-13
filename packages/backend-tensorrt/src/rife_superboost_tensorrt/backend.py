# SPDX-License-Identifier: MIT
"""Optional TensorRT configuration and per-node C ABI session ownership."""

import ctypes as C
from dataclasses import dataclass, replace, field
import json
import os
from pathlib import Path
import threading
import weakref

from rife_superboost._protocol import FrameView
from rife_superboost.errors import InferenceError, RifeError
from rife_superboost.models import Model

from ._cache import ensure_engine
from ._loader import load_native, resolve_runtime
from ._progress import BuildFeedback


def _destroy(native, handle, lock) -> None:
    """Finalize one worker after requests release their owning session.

    Args:
        native: Strongly held DLL owner.
        handle: Opaque live C ABI worker pointer.
        lock: Per-session lock also used by explicit close and execution.
    """
    with lock:
        native.dll.rsbDestroy(handle)


class TensorRTSession:
    """Node-owned full/split contexts, one stream and bounded frame/staging buffers."""

    def __init__(self, native, engines: dict, model: Model, width: int, height: int,
                 config: "TensorRT", diagnostics: dict) -> None:
        """Args:
            native: Process-retained native DLL owner.
            engines: Hash-checked full engine or encoder/synthesis pair.
            model: Prepared profile and channel contract.
            width: Active source width before padding.
            height: Active source height before padding.
            config: Owning CUDA device and bounded execution options.
            diagnostics: Cache and loaded-device identities for inspection.
        """
        self.native, self.width, self.height = native, width, height
        self.diagnostics = diagnostics
        self.reuse_encoding = config.encoding_cache
        self.lock = threading.RLock()
        handle = C.c_void_p()
        native.check(native.dll.rsbCreateV2(str(engines.get("full", engines.get("synthesis"))),
                     str(engines["encoder"]) if "encoder" in engines else None,
                     str(engines["synthesis"]) if "synthesis" in engines else None,
                     width, height, model.alignment, model.encoding_channels, config.device,
                     config.frame_cache_mib * 1024 * 1024, config.cache_slots, config.cuda_graph, C.byref(handle)))
        self.handle = handle
        self.finalizer = weakref.finalize(self, _destroy, native, handle, self.lock)

    def run(self, left: FrameView, right: FrameView, output: FrameView, timestep: float,
            *, frame_ids: tuple[int, int] | None = None, enabled: bool = True) -> None:
        """Execute a synchronous request with borrowed, stride-aware RGB planes.

        Args:
            left: First source frame, kept alive until completion.
            right: Second source frame with matching dimensions and sample type.
            output: Writable output frame owned by the request.
            timestep: Runtime scalar time; never frozen into the engine.
            frame_ids: Immutable IDs scoped to this session. None disables reuse.
                Call reset() before IDs refer to changed content.
            enabled: False copies the left frame on CPU, preserving the GPU cache.
        """
        frames = (left, right, output)
        if type(enabled) is not bool:
            raise InferenceError("enabled must be a per-request boolean")
        if any((f.width, f.height, f.sample_bytes) != (self.width, self.height, left.sample_bytes) for f in frames) or left.sample_bytes not in (2, 4):
            raise InferenceError("Frame dimensions or RGB sample types do not match this session")
        pointers = [(C.c_void_p * 3)(*frame.planes) for frame in frames]
        strides = [(C.c_int64 * 3)(*frame.strides) for frame in frames]
        if frame_ids is not None and (not isinstance(frame_ids, tuple) or len(frame_ids) != 2 or
                any(type(i) is not int or not 0 <= i < 2**64 for i in frame_ids)):
            raise InferenceError("frame_ids must be two immutable nonnegative uint64 identities")
        if not self.reuse_encoding:
            frame_ids = None
        ids = (0, 1) if frame_ids is None else frame_ids
        with self.lock:
            if not self.finalizer.alive:
                raise InferenceError("TensorRT session is closed")
            self.native.check(self.native.dll.rsbRunFrames(self.handle, *ids, frame_ids is not None, enabled, timestep, left.sample_bytes,
                              pointers[0], strides[0], pointers[1], strides[1], pointers[2], strides[2]))

    def reset(self) -> None:
        """Invalidate source content identities after all current requests finish."""
        with self.lock:
            if not self.finalizer.alive:
                raise InferenceError("TensorRT session is closed")
            self.native.check(self.native.dll.rsbReset(self.handle))

    def statistics(self) -> dict:
        """Return cache counters and allocation; budget excludes engine/workspace/staging."""
        with self.lock:
            if not self.finalizer.alive:
                raise InferenceError("TensorRT session is closed")
            data = C.create_string_buffer(4096)
            self.native.check(self.native.dll.rsbStats(self.handle, data, len(data)))
            return json.loads(data.value)

    def close(self) -> None:
        """Release this worker after pending synchronous requests finish."""
        self.finalizer()


@dataclass(frozen=True)
class TensorRT:
    """Optional native backend; construction does not load NVIDIA libraries.

    Args:
        runtime_dir: NVIDIA DLL directory. None selects the pinned official library wheel.
        cache_dir: Writable cache root. None selects LOCALAPPDATA/RIFE-SuperBoost/tensorrt.
        device: CUDA device ordinal.
        workspace_mib: Engine-builder workspace limit in MiB.
        build_timeout: Maximum native-builder wall time in seconds.
        encoding_cache: Reuse RGB/features by frame ID. False selects a supplied full
            development graph, or recomputes both encoders with split-only packages.
        frame_cache_mib: Hard RGB/feature cache allocation budget, excluding engines,
            context workspace, fixed synthesis buffers, and transfer staging.
        cache_slots: Additional hard slot cap; at least two slots must fit.
        cuda_graph: Capture encoder slots and fixed-binding synthesis pairs; the
            synthesis Graph LRU is bounded to sixteen entries.
        precision: Convolution arithmetic policy, fp16 by default. Prepared mixed
            graphs retain FP32 residual accumulation/sampling/geometry, independent
            of RGBH/RGBS IO. Neither policy converts graphs during playback.
        build_feedback: console shows cold-build/wait status in one system terminal
            for all engine roles; log is headless. window remains an alias.
        fused: Use supplied TensorRT fusion assets when the model package has them.
        io_precision: auto preserves RGBH/RGBS source precision; explicit fp16/fp32
            selects prepared storage. RGBS is not implicitly rounded to RGBH.

    Both policies disable TF32. Geometry remains FP32; prepared fused layouts
    retain RGBH/RGBS source storage and the encoder's actual output precision.
    """

    runtime_dir: str | Path | None = None
    cache_dir: str | Path | None = None
    device: int = 0
    workspace_mib: int = 512
    build_timeout: float = 900
    encoding_cache: bool = True
    frame_cache_mib: int = 256
    cache_slots: int = 4
    cuda_graph: bool = False
    precision: str = "fp16"
    build_feedback: str = "console"
    fused: bool = True
    io_precision: str = "auto"
    _input_bytes: int = field(default=4, repr=False)

    def for_input(self, sample_bytes: int) -> "TensorRT":
        """Args:
            sample_bytes: VS RGBH/RGBS sample width; raw callers can bind it too.
        """
        if type(sample_bytes) is not int or sample_bytes not in (2, 4):
            raise RifeError("Input samples must be two or four bytes")
        return replace(self, _input_bytes=sample_bytes)

    @staticmethod
    def capabilities() -> dict:
        """Report implemented policies without loading NVIDIA libraries."""
        return {"platform": "windows-x64", "network_precisions": ["fp16", "fp32"],
                "default_precision": "fp16", "geometry": "fp32", "graph_io_storage": ["fp16", "fp32"],
                "prepared_fusions": True, "direct_cache_binding": True, "build_feedback": ["console", "log"],
                "formats": ["RGBH", "RGBS"], "request_skip": True, "encoding_cache": True,
                "cuda_graph": True, "color_conversion": False}

    def create_session(self, model: Model, width: int, height: int) -> TensorRTSession:
        """Create a real native session from an already prepared graph.

        Args:
            model: Verified reference or pre-optimized full/split graph descriptor.
            width: Original frame width, including odd sizes.
            height: Original frame height, including odd sizes.
        """
        if type(self.device) is not int or self.device < 0:
            raise RifeError("device must be a nonnegative integer")
        if type(self.workspace_mib) is not int or not 64 <= self.workspace_mib <= 4096:
            raise RifeError("workspace_mib must be an integer in 64..4096")
        if not isinstance(self.build_timeout, (int, float)) or not 1 <= self.build_timeout <= 86400:
            raise RifeError("build_timeout must be 1..86400 seconds")
        if type(width) is not int or type(height) is not int or not 1 <= width <= 8192 or not 1 <= height <= 8192:
            raise RifeError("Frame width and height must be integers in 1..8192")
        if not isinstance(self.precision, str) or self.precision not in ("fp16", "fp32"):
            raise RifeError("precision must be 'fp16' or 'fp32'")
        if self.build_feedback not in ("window", "console", "log"):
            raise RifeError("build_feedback must be 'console' or 'log'")
        model.verify()
        if any(type(v) is not bool for v in (self.encoding_cache, self.cuda_graph, self.fused)):
            raise RifeError("encoding_cache, cuda_graph and fused must be booleans")
        if self.io_precision not in ("auto", "fp16", "fp32"):
            raise RifeError("io_precision must be auto, fp16 or fp32")
        if type(self.frame_cache_mib) is not int or not 1 <= self.frame_cache_mib <= 4096:
            raise RifeError("frame_cache_mib must be an integer in 1..4096")
        if type(self.cache_slots) is not int or not 2 <= self.cache_slots <= 32:
            raise RifeError("cache_slots must be an integer in 2..32")
        if model.alignment != 128:
            raise RifeError("The validated official reference alignment is 128")
        if model.quantization != "none":
            raise RifeError("This TensorRT backend does not support PTQ/QAT variants; keep their prepared precision policy")
        if model.validated_backends and "tensorrt" not in model.validated_backends:
            raise RifeError("This prepared graph family has not been validated for TensorRT")
        storage = ("fp16" if self._input_bytes == 2 else "fp32") if self.io_precision == "auto" else self.io_precision
        model = model.for_backend(self.precision, storage) if self.fused and model.backend_variants else model.for_precision(self.precision)
        split = model.encoder is not None and (self.encoding_cache or model.graph is None)
        if split:
            pw, ph = (width + 127) // 128 * 128, (height + 127) // 128 * 128
            minimum = pw * ph * (3 * (2 if model.io_precision == "fp16" else 4) + model.encoding_channels * (2 if model.feature_precision == "fp16" else 4)) * 2
            if self.frame_cache_mib * 1024 * 1024 < minimum:
                raise RifeError(f"frame_cache_mib cannot hold two RGB/feature slots; need at least {(minimum + 1048575) // 1048576} MiB")
        runtime_dir = resolve_runtime(self.runtime_dir)
        if self.cache_dir:
            cache_dir = Path(self.cache_dir).expanduser().resolve()
        elif os.environ.get("LOCALAPPDATA"):
            cache_dir = Path(os.environ["LOCALAPPDATA"]) / "RIFE-SuperBoost/tensorrt"
        else:
            raise RifeError("A writable cache_dir or LOCALAPPDATA is required")
        native = load_native(runtime_dir)
        header = f"Model: {model.model_id} | {width}x{height} | {model.precision}"
        with BuildFeedback(native, cache_dir, header, self.build_feedback != "log", False) as progress:
            engines, reports = {}, {}
            roles = ("encoder", "synthesis") if split else ("full",)
            for index, role in enumerate(roles, 1):
                progress.set_stage(index, len(roles))
                engines[role], reports[role] = ensure_engine(native, model, width, height, self.device, self.workspace_mib,
                                                            cache_dir, float(self.build_timeout), role, self.build_feedback, progress)
            diagnostics = dict(reports.get("full", reports.get("synthesis")))
            diagnostics.update(engines=reports, encoding_cache=split and self.encoding_cache,
                               execution_layout="split" if split else "full", cuda_graph=self.cuda_graph,
                               variant_id=model.variant_id, weight_family=model.weight_family, quantization=model.quantization)
            diagnostics.update(precision=model.precision, geometry="fp32", graph_io_storage=model.io_precision,
                               feature_storage=model.feature_precision, prepared_execution=model.execution)
            diagnostics["cache_hit"] = all(report["cache_hit"] for report in reports.values())
            if progress.active:
                progress.update("building", "Loading compiled engines; video processing will resume shortly")
            return TensorRTSession(native, engines, model, width, height, self, diagnostics)
