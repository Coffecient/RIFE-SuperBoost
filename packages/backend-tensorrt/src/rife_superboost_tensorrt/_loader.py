# SPDX-License-Identifier: MIT
"""Explicit native loading with retained handles and loaded-DLL conflict checks."""

import ctypes as C
from ctypes import wintypes as W
from importlib.metadata import distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import struct
import threading

from rife_superboost.errors import DependencyError, InferenceError, VersionConflictError
from rife_superboost.models import file_digest

_lock = threading.RLock()
_loaded = {}
EXPECTED_VERSION = (10, 16, 1, 11)
NVIDIA_NAMES = ("nvinfer_10.dll", "nvonnxparser_10.dll", "nvinfer_plugin_10.dll",
                "nvinfer_lean_10.dll", "nvinfer_dispatch_10.dll", "nvinfer_vc_plugin_10.dll",
                "nvinfer_builder_resource_ptx_10.dll", *(f"nvinfer_builder_resource_sm{sm}_10.dll" for sm in (75, 80, 86, 89, 90, 100, 120)))


def resolve_runtime(runtime_dir: str | Path | None = None) -> Path:
    """Locate NVIDIA's library distribution without executing its eager initializer.

    Args:
        runtime_dir: Explicit DLL directory. None requires the pinned official wheel.
    """
    if runtime_dir is not None:
        return Path(runtime_dir).expanduser().resolve()
    try:
        package = distribution("tensorrt-cu13-libs")
    except PackageNotFoundError as error:
        raise DependencyError("NVIDIA libraries are missing. Install tensorrt-cu13-libs==10.16.1.11 "
                              "from https://pypi.nvidia.com or supply TensorRT(runtime_dir=...).") from error
    if package.version != ".".join(map(str, EXPECTED_VERSION)):
        raise VersionConflictError(f"Expected tensorrt-cu13-libs==10.16.1.11; installed={package.version}. Restart after changing libraries.")
    folder = Path(package.locate_file("tensorrt_libs")).resolve()
    if not folder.is_dir():
        raise DependencyError(f"The installed NVIDIA distribution has no runtime directory: {folder}")
    return folder


def module_path(name: str) -> Path | None:
    """Find an already loaded module without loading it.

    Args:
        name: Windows DLL basename.
    """
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [W.LPCWSTR]
    kernel.GetModuleHandleW.restype = W.HMODULE
    kernel.GetModuleFileNameW.argtypes = [W.HMODULE, W.LPWSTR, W.DWORD]
    kernel.GetModuleFileNameW.restype = W.DWORD
    handle = kernel.GetModuleHandleW(name)
    if not handle:
        return None
    buffer = C.create_unicode_buffer(32768)
    length = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
    if not length or length >= len(buffer):
        raise DependencyError(f"Cannot inspect loaded module {name}")
    return Path(buffer.value).resolve()


def file_version(path: Path) -> tuple[int, int, int, int] | None:
    """Read an optional Windows file version; NVIDIA DLLs may omit this resource.

    Args:
        path: DLL with a VS_FIXEDFILEINFO resource.
    """
    version = C.WinDLL("version", use_last_error=True)
    version.GetFileVersionInfoSizeW.argtypes = [W.LPCWSTR, C.POINTER(W.DWORD)]
    version.GetFileVersionInfoSizeW.restype = W.DWORD
    version.GetFileVersionInfoW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p]
    version.GetFileVersionInfoW.restype = W.BOOL
    version.VerQueryValueW.argtypes = [C.c_void_p, W.LPCWSTR, C.POINTER(C.c_void_p), C.POINTER(W.UINT)]
    version.VerQueryValueW.restype = W.BOOL
    ignored = W.DWORD()
    size = version.GetFileVersionInfoSizeW(str(path), C.byref(ignored))
    if not size:
        return None
    data = C.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, data):
        raise DependencyError(f"Cannot read DLL version data: {path}")
    pointer, length = C.c_void_p(), W.UINT()
    if not version.VerQueryValueW(data, "\\", C.byref(pointer), C.byref(length)) or length.value < 16:
        raise DependencyError(f"Cannot read fixed DLL version: {path}")
    values = C.cast(pointer, C.POINTER(W.DWORD))
    high, low = values[2], values[3]
    return high >> 16, high & 65535, low >> 16, low & 65535


def require_x64(path: Path) -> None:
    """Reject a missing or non-x64 native image before LoadLibrary.

    Args:
        path: PE executable or DLL expected to target AMD64.
    """
    if not path.is_file():
        raise DependencyError(f"Native dependency is missing: {path}")
    try:
        with path.open("rb") as stream:
            if stream.read(2) != b"MZ":
                raise VersionConflictError(f"Not a Windows PE image: {path}")
            stream.seek(0x3C)
            offset = struct.unpack("<I", stream.read(4))[0]
            stream.seek(offset)
            if stream.read(4) != b"PE\0\0" or struct.unpack("<H", stream.read(2))[0] != 0x8664:
                raise VersionConflictError(f"An x64 PE image is required: {path}")
    except (OSError, struct.error) as error:
        raise VersionConflictError(f"Cannot inspect x64 PE image {path}: {error}") from error


class Native:
    """Process-retained DLL handles; live sessions keep a strong reference too."""

    def __init__(self, runtime_dir: Path) -> None:
        """Args:
            runtime_dir: Explicit directory or resolved official NVIDIA library wheel.
        """
        if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
            raise DependencyError("The TensorRT backend supports Windows x64 only")
        self.folder = Path(__file__).resolve().parent / "_native"
        self.runtime_dir = runtime_dir.resolve()
        self.binary = self.folder / "rsb_runtime.dll"
        self.builder = self.folder / "rsb_builder.exe"
        for path in (self.binary, self.builder, self.folder / "rsb_progress.exe", self.runtime_dir / "nvinfer_10.dll", self.runtime_dir / "nvonnxparser_10.dll"):
            require_x64(path)
        for name in ("nvinfer_10.dll", "nvonnxparser_10.dll"):
            path = self.runtime_dir / name
            actual = file_version(path)
            if actual is not None and actual != EXPECTED_VERSION:
                raise VersionConflictError(f"Expected NVIDIA file version {EXPECTED_VERSION}; found {actual}: {path}")
        self.check_conflicts()
        self.directories = [os.add_dll_directory(str(path)) for path in (self.folder, self.runtime_dir)]
        try:
            # Load the principal runtime by absolute path before resolving our DLL imports.
            self.nvidia = C.WinDLL(str(self.runtime_dir / "nvinfer_10.dll"), winmode=0x1100)
            actual_version = []
            for part in ("Major", "Minor", "Patch", "Build"):
                function = getattr(self.nvidia, f"getInferLib{part}Version")
                function.argtypes = []
                function.restype = C.c_int32
                actual_version.append(function())
            if tuple(actual_version) != EXPECTED_VERSION:
                raise VersionConflictError(f"Expected TensorRT {EXPECTED_VERSION}; loaded {tuple(actual_version)} from {self.runtime_dir}")
            self.dll = C.CDLL(str(self.binary), winmode=0x1100)
        except AttributeError as error:
            raise VersionConflictError(f"The NVIDIA runtime lacks the required TensorRT version-query ABI: {self.runtime_dir}; {error}") from error
        except OSError as error:
            cause = "Missing transitive DLL" if getattr(error, "winerror", None) == 126 else "Native dependency load failed"
            raise DependencyError(f"{cause} ({error}); own DLL={self.binary}; NVIDIA directory={self.runtime_dir}. Check transitive dependencies and process architecture.") from error
        self.check_conflicts()
        d = self.dll
        d.rsbAbi.argtypes = []; d.rsbAbi.restype = C.c_int
        if d.rsbAbi() != 3:
            raise VersionConflictError(f"Unsupported RIFE-SuperBoost C ABI {d.rsbAbi()}; expected 3: {self.binary}")
        d.rsbError.argtypes = []; d.rsbError.restype = C.c_char_p
        d.rsbDevice.argtypes = [C.c_int, C.c_void_p, C.c_size_t]; d.rsbDevice.restype = C.c_int
        d.rsbCreateV2.argtypes = [C.c_wchar_p, C.c_wchar_p, C.c_wchar_p, C.c_int, C.c_int,
                                 C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_int, C.c_int, C.POINTER(C.c_void_p)]
        d.rsbCreateV2.restype = C.c_int
        pointers, strides = C.POINTER(C.c_void_p), C.POINTER(C.c_int64)
        d.rsbRun.argtypes = [C.c_void_p, C.c_float, C.c_int, pointers, strides, pointers, strides, pointers, strides]
        d.rsbRun.restype = C.c_int
        d.rsbRunFrames.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_int, C.c_int, C.c_float, C.c_int,
                                   pointers, strides, pointers, strides, pointers, strides]
        d.rsbRunFrames.restype = C.c_int
        d.rsbReset.argtypes = [C.c_void_p]; d.rsbReset.restype = C.c_int
        d.rsbStats.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]; d.rsbStats.restype = C.c_int
        d.rsbDestroy.argtypes = [C.c_void_p]; d.rsbDestroy.restype = C.c_int
        self.identity = {"abi": 3, "runtime_sha256": file_digest(self.binary), "builder_sha256": file_digest(self.builder),
                         "nvidia_version": list(EXPECTED_VERSION),
                         "nvidia_sha256": {name: file_digest(self.runtime_dir / name) for name in ("nvinfer_10.dll", "nvonnxparser_10.dll")}}

    def check_conflicts(self) -> None:
        """Require exact file identity for a same-name DLL already in this process."""
        for name in (*NVIDIA_NAMES, "rsb_runtime.dll"):
            actual = module_path(name)
            expected = self.binary if name == "rsb_runtime.dll" else self.runtime_dir / name
            if actual and actual != expected.resolve():
                if not expected.is_file() or file_digest(actual) != file_digest(expected):
                    raise VersionConflictError(f"Conflicting loaded DLL: {name}; loaded={actual}; requested={expected}. Restart the host with compatible libraries.")

    def check(self, success: int) -> None:
        """Raise the native thread's current diagnostic.

        Args:
            success: C ABI return code, one on success.
        """
        if not success:
            raise InferenceError(self.dll.rsbError().decode("utf-8", "replace"))

    def device(self, ordinal: int) -> dict:
        """Query actual device, driver, CUDA runtime, and TensorRT module identity.

        Args:
            ordinal: Selected CUDA device ordinal.
        """
        data = C.create_string_buffer(16384)
        self.check(self.dll.rsbDevice(ordinal, data, len(data)))
        return json.loads(data.value)


def load_native(runtime_dir: Path) -> Native:
    """Reuse native libraries while keeping Python DLL-directory handles alive.

    Args:
        runtime_dir: Explicit or installed NVIDIA directory; never guessed from PATH.
    """
    with _lock:
        key = str(runtime_dir.resolve())
        if key not in _loaded:
            _loaded[key] = Native(runtime_dir)
        native = _loaded[key]
        if file_digest(native.binary) != native.identity["runtime_sha256"]:
            raise VersionConflictError("The loaded runtime DLL changed on disk; restart the host")
        if file_digest(native.builder) != native.identity["builder_sha256"] or any(
                file_digest(native.runtime_dir / name) != digest for name, digest in native.identity["nvidia_sha256"].items()):
            raise VersionConflictError("Loaded NVIDIA libraries or the native builder changed on disk; restart the host")
        native.check_conflicts()
        return native
