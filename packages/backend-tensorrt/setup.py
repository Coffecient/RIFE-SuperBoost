# SPDX-License-Identifier: MIT
"""Tag the C ABI binaries as Windows x64 without tying them to a CPython ABI."""

from pathlib import Path
import hashlib
import json
from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        """Mark this wheel as containing native platform code."""
        return True


class PreparedBuild(build_py):
    def run(self):
        """Fail closed when building a source tree without its staged native assets."""
        folder = Path(__file__).parent / "src/rife_superboost_tensorrt/_native"
        for name in ("rsb_runtime.dll", "rsb_builder.exe", "rsb_progress.exe"):
            if not (folder / name).is_file():
                raise RuntimeError(f"Build native binaries before building this wheel: {name}")
        info = json.loads((folder / "build-info.json").read_text(encoding="utf-8-sig"))
        if info["tensorrt"] != "10.16.1.11" or info["cuda"] != "13.2":
            raise RuntimeError("Native build metadata does not match the pinned SDKs")
        for name, key in (("rsb_runtime.dll", "runtime_sha256"), ("rsb_builder.exe", "builder_sha256"), ("rsb_progress.exe", "progress_sha256")):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != info[key]:
                raise RuntimeError("Staged native binary does not match build-info.json: " + name)
        for name in ("CUDA-EULA.txt", "CUDA-LICENSE.txt", "NOTICE.txt"):
            if not (folder.parent / "licenses" / name).is_file():
                raise RuntimeError("Missing native redistribution notice: " + name)
        super().run()


class NativeWheel(bdist_wheel):
    def get_tag(self):
        """The DLL uses ctypes and has no CPython extension ABI."""
        return "py3", "none", "win_amd64"

setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": NativeWheel, "build_py": PreparedBuild})
