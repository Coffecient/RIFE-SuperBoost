# SPDX-License-Identifier: MIT
"""Locked, isolated native engine builds and atomic cache publication."""

from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import uuid

from rife_superboost.errors import EngineBuildError
from rife_superboost.models import Model, file_digest
from ._progress import BuildFeedback


def atomic_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file using an atomic same-directory replacement.

    Args:
        path: Destination in a writable user cache.
        value: Serializable manifest with finite numbers.
    """
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def build_lock(path: Path, timeout: float):
    """Acquire a kernel-released Windows byte-range lock across build processes.

    Args:
        path: Persistent lock file belonging to one cache identity.
        timeout: Maximum seconds to wait for an existing build.
    """
    import msvcrt

    with path.open("a+b") as stream:
        deadline = time.monotonic() + timeout
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise EngineBuildError(f"Timed out waiting for engine build lock: {path}") from error
                time.sleep(0.1)
        try:
            # Windows permits locking beyond EOF. Initialize only after acquisition:
            # an unlocked concurrent append can otherwise fail against the owner.
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def cache_valid(folder: Path, identity: dict) -> bool:
    """Accept only a complete, exact-identity, hash-checked engine publication.

    Args:
        folder: One cache entry directory.
        identity: Expected graph, native, hardware, precision, and shape identity.
    """
    try:
        manifest = json.loads((folder / "engine.json").read_text(encoding="utf-8"))
        return (manifest["identity"] == identity and manifest["status"] == "ready" and
                set(manifest["files"]) == {"model.engine", "timing.cache"} and
                all(file_digest(folder / name) == digest for name, digest in manifest["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_engine(native, model: Model, width: int, height: int, device: int, workspace: int,
                  cache_dir: Path, timeout: float, role: str = "full", feedback: str = "log",
                  progress: BuildFeedback | None = None) -> tuple[Path, dict]:
    """Compile the supplied graph verbatim, or reuse its validated device cache.

    Args:
        native: Loaded native runtime retaining private NVIDIA directory handles.
        model: Verified prepared graph; never rewritten or split here.
        width: Original source width.
        height: Original source height.
        device: CUDA ordinal.
        workspace: TensorRT workspace budget in MiB.
        cache_dir: Explicit or platform user cache, outside model package data.
        timeout: Build-process timeout in seconds.
        role: Pre-exported full, encoder or synthesis graph; never split here.
        feedback: console opens a native terminal on cold builds; log is headless.
        progress: Optional session-owned observer shared by encoder and synthesis.
    """
    info = native.device(device)
    graph, digest = model.asset(role)
    if graph is None:
        raise EngineBuildError(f"The model has no prepared {role} graph")
    alignment = model.alignment
    padded_width = (width + alignment - 1) // alignment * alignment
    padded_height = (height + alignment - 1) // alignment * alignment
    major, minor = info["compute_capability"]
    resources = list(native.runtime_dir.glob("nvinfer_builder_resource_*_10.dll"))
    if not (native.runtime_dir / f"nvinfer_builder_resource_sm{major}{minor}_10.dll").is_file() and not (native.runtime_dir / "nvinfer_builder_resource_ptx_10.dll").is_file():
        raise EngineBuildError(f"Builder resource DLL for SM{major}{minor} is absent: {native.runtime_dir}")
    # Cache these immutable SDK hashes after their first use in this process.
    if not hasattr(native, "resource_identity"):
        native.resource_identity = {path.name: file_digest(path) for path in sorted(resources)}
    identity = {"cache_schema": 5, "model_id": model.model_id, "graph_sha256": digest, "role": role,
                "variant_id": model.variant_id, "weight_family": model.weight_family, "quantization": model.quantization,
                "external_data_sha256": sorted(digest for _, digest in model.external_data),
                "encoding_channels": model.encoding_channels, "graph_contract": model.contract,
                "padding": f"right-bottom-zero-{alignment}",
                "shape": [1, 3, padded_height, padded_width], "native": native.identity,
                "builder_resources": native.resource_identity,
                "device": {key: value for key, value in info.items() if key != "tensorrt_path"},
                "build": {"precision": model.precision, "geometry": "fp32", "tf32": False, "workspace_mib": workspace,
                          "optimization_level": 3, "aux_streams": 0,
                          "io_precision": model.io_precision, "feature_precision": model.feature_precision,
                          "execution": model.execution, "plugins": ["rsb-fusions-v1"] if model.execution == "trt-fused-v1" else []}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    folder = cache_dir.resolve() / "v1" / key
    folder.mkdir(parents=True, exist_ok=True)
    hit = False
    message = f"Model: {model.model_id} | {role} | {width}x{height} -> {padded_width}x{padded_height} | {model.precision}\nCache: {folder}"
    own_progress = progress is None
    progress = progress or BuildFeedback(native, folder, message, feedback in ("console", "window"), False)
    with (progress if own_progress else nullcontext(progress)):
        if not cache_valid(folder, identity):
            progress.begin(message)
        with build_lock(folder / "build.lock", timeout + 60):
            if cache_valid(folder, identity):
                hit = True
                if progress.active:
                    progress.begin(message, "Reusing the validated cached engine")
            else:
                progress.begin(message)
                job = folder / ("job-" + uuid.uuid4().hex)
                job.mkdir()
                engine, timing = job / "model.engine", job / "timing.cache"
                previous_timing = folder / "timing.cache"
                if previous_timing.is_file():
                    # A timing cache is still checked by TensorRT's own compatibility header.
                    timing.write_bytes(previous_timing.read_bytes())
                command = [str(native.builder), "--device", str(device), "--runtime-dir", str(native.runtime_dir),
                           "--onnx", str(graph), "--engine", str(engine), "--timing", str(timing),
                           "--width", str(padded_width), "--height", str(padded_height), "--workspace-mib", str(workspace),
                           "--alignment", str(alignment), "--role", role, "--channels", str(model.encoding_channels),
                           "--precision", model.precision, "--io-precision", model.io_precision,
                           "--feature-precision", model.feature_precision]
                atomic_json(job / "request.json", {"identity": identity, "command": command})
                env = dict(os.environ)
                system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
                env["PATH"] = os.pathsep.join(map(str, (native.runtime_dir, native.folder, system / "System32", system)))
                log = job / "build.log"
                progress.update("building", "Parsing the model and selecting GPU tactics; this can take several minutes", log)
                logging.getLogger("rife_superboost").info("Building TensorRT engine; log=%s", log)
                try:
                    with log.open("wb") as stream:
                        result = subprocess.run(command, cwd=job, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                                timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise EngineBuildError(f"Native builder could not complete: {error}; log={log}") from error
                if result.returncode or not engine.is_file() or not timing.is_file():
                    raise EngineBuildError(f"Native engine build failed with exit code {result.returncode}; precision={model.precision}; "
                                           f"workspace_mib={workspace}; log={log}; NVIDIA directory={native.runtime_dir}")
                try:
                    outcome = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
                    actual_runtime = Path(outcome["device"]["tensorrt_path"]).resolve()
                    if actual_runtime != (native.runtime_dir / "nvinfer_10.dll").resolve():
                        raise ValueError("Builder runtime origin does not match the requested private SDK")
                except (ValueError, KeyError, IndexError) as error:
                    raise EngineBuildError(f"Invalid builder result: {error}; log={log}") from error
                # Publish data first and the ready marker last, while holding the lock.
                files = {name: file_digest(job / name) for name in ("model.engine", "timing.cache")}
                for name in files:
                    os.replace(job / name, folder / name)
                atomic_json(folder / "engine.json", {"status": "ready", "identity": identity, "files": files,
                                                     "builder_result": outcome, "log": str(log)})
                if not cache_valid(folder, identity):
                    raise EngineBuildError(f"Atomic engine publication failed validation: {folder}")
    return folder / "model.engine", {"cache_hit": hit, "folder": str(folder), "device": info,
                                      "identity": identity, "manifest": str(folder / "engine.json")}
