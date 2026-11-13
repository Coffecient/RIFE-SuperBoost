# SPDX-License-Identifier: MIT
"""Refuse an incomplete data wheel; all optimization occurs before this hook."""

import hashlib
import json
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class PreparedBuild(build_py):
    def run(self):
        """Verify the complete pre-generated asset set using only the standard library."""
        folder = next((Path(__file__).parent / "src").glob("rife_superboost_model_*")) / "assets"
        try:
            data = json.loads((folder / "model.json").read_text(encoding="utf-8"))
            if data["schema_version"] != 2 or data["layout"] not in ("split-shared-float-v1", "split-shared-float-v2"):
                raise ValueError("Unsupported prepared distribution layout")
            assets = list(data["external_data"])
            entries = [data, data["variants"]["fp16"]]
            if data["layout"] == "split-shared-float-v2":
                variants = data["backend_variants"]
                if len(variants) != 4 or any(v["execution"] != "trt-fused-v1" for v in variants.values()):
                    raise ValueError("Incomplete prepared TensorRT layout family")
                entries.extend(variants.values())
            for entry in entries:
                if "graph" in entry or set(entry["cache_graphs"]) != {"encoder", "synthesis"}:
                    raise ValueError("User packages require split graphs only")
                assets.extend(entry["cache_graphs"].values())
            for asset in assets:
                path = (folder / asset["file"]).resolve()
                if not path.is_relative_to(folder.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                    raise ValueError("Invalid prepared model asset: " + asset["file"])
            for name in ("LICENSE.Practical-RIFE", "provenance.json"):
                if not (folder / name).is_file():
                    raise ValueError("Missing model notice/provenance: " + name)
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError("Run tools/prepare-release-models.py before building model wheels: " + str(error)) from error
        super().run()


setup(cmdclass={"build_py": PreparedBuild})
