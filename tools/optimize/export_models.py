# SPDX-License-Identifier: MIT
"""Generate optimized, dynamic release assets before installation or playback."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import onnx
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/export"))
from assets import build_reference, read_models, sha256, write_json
from export_reference import environment
from network import Optimized, Passes

VERSION = "m3-dynamic-v1"


def export(model_id: str, output: Path, passes: Passes) -> dict:
    """Args:
        model_id: One of the four pinned official model IDs.
        output: Development/release output directory within this project.
        passes: Explicit graph transforms recorded in every asset's metadata.
    """
    model = read_models()[model_id]
    reference_path = ROOT / f"build/reference/rife-{model_id}-reference-fp32.onnx"
    references = json.loads((ROOT / "models/reference-manifest.json").read_text(encoding="utf-8"))
    # Verify M1 identity before deriving assets from its exact network parameters.
    recorded = next(item for item in references["models"] if item["id"] == model_id)
    if sha256(reference_path) != recorded["artifact"]["sha256"]:
        raise ValueError("M1 reference graph changed")
    reference, parameters = build_reference(model)
    optimized = Optimized(reference, passes).eval().requires_grad_(False)
    folder = output / model_id
    folder.mkdir(parents=True, exist_ok=True)
    image0, image1 = torch.zeros(1, 3, 128, 256), torch.ones(1, 3, 128, 256)
    timestep = torch.full((1, 1, 1, 1), 0.5)
    with torch.inference_mode():
        feature0, feature1 = optimized.encode(image0), optimized.encode(image1)
        entries = {"full": (optimized, (image0, image1, timestep), ["image0", "image1", "timestep"], "output"),
                   "encoder": (optimized.encode, (image0,), ["image"], "feature"),
                   "synthesis": (optimized.synthesis, (image0, image1, feature0, feature1, timestep),
                                 ["image0", "image1", "feature0", "feature1", "timestep"], "output")}
        assets = {}
        for role, (module, inputs, names, result) in entries.items():
            path = folder / f"{role}.onnx"
            axes = {name: {2: "height", 3: "width"} for name in [*names, result] if name != "timestep"}
            torch.onnx.export(module, inputs, str(path), input_names=names, output_names=[result],
                              dynamic_axes=axes, opset_version=16, dynamo=False, do_constant_folding=True,
                              external_data=False)
            graph = onnx.load(path)
            onnx.helper.set_model_props(graph, {"project": "RIFE-SuperBoost", "version": VERSION,
                "role": role, "model_id": model_id, "passes": json.dumps(asdict(passes), sort_keys=True),
                "reference_sha256": sha256(reference_path), "license": "MIT; Copyright (c) 2021 hzwer"})
            onnx.checker.check_model(graph, full_check=True)
            onnx.save(graph, path)
            assets[role] = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                            "operators": dict(sorted(Counter(n.op_type for n in graph.graph.node).items()))}
    descriptor = {"schema_version": 1, "id": model_id, "purpose": "production", "alignment": 128,
                  "graph": {k: assets["full"][k] for k in ("file", "sha256")},
                  "cache_graphs": {role: {k: assets[role][k] for k in ("file", "sha256")}
                                   for role in ("encoder", "synthesis")},
                  "encoding_channels": model["reference_profile"]["encoding_channels"],
                  "contract": VERSION, "precision": "fp32", "spatial": "dynamic-aligned-hw",
                  "scales": model["reference_profile"]["stage_scale_denominators"],
                  "passes": asdict(passes), "reference_sha256": sha256(reference_path)}
    provenance = {"official_source": model["official_source"], "archive_sha256": model["archive"]["sha256"],
                  "checkpoint_sha256": model["checkpoint"]["sha256"], "source_files": model["source_files"],
                  "license": "MIT", "copyright": "Copyright (c) 2021 hzwer",
                  "reference_profile": model["reference_profile"], "environment": environment()}
    provenance["official_source"] = {**provenance["official_source"], "license_file": "LICENSE.Practical-RIFE"}
    shutil.copyfile(ROOT / "models/licenses/practical-rife/LICENSE", folder / "LICENSE.Practical-RIFE")
    descriptor["license_file"] = "LICENSE.Practical-RIFE"
    write_json(folder / "provenance.json", provenance)
    write_json(folder / "model.json", descriptor)
    report = {"model_id": model_id, "descriptor": (folder / "model.json").relative_to(ROOT).as_posix(),
              "reference_sha256": sha256(reference_path), "passes": asdict(passes), "assets": assets,
              "encoding_channels": descriptor["encoding_channels"], "scales": descriptor["scales"],
              "parameters": parameters, "environment": environment(),
              "provenance_sha256": sha256(folder / "provenance.json"),
              "exporter_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("network.py"),
                                   ROOT / "tools/export/reference.py", ROOT / "tools/export/assets.py")}}
    write_json(folder / "export.json", report)
    print(f"Exported {model_id}: dynamic full/encoder/synthesis", flush=True)
    return report


def main() -> None:
    """Export generic ONNX assets; never called by an installed runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(read_models()), choices=list(read_models()), help="Pinned model IDs.")
    parser.add_argument("--output", type=Path, default=ROOT / "build/optimized", help="Release asset staging directory.")
    parser.add_argument("--no-repack", action="store_true", help="Keep original deconvolution operators for ablation.")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must stay within this project")
    torch.set_num_threads(4)
    passes = Passes(repack=not args.no_repack)
    reports = [export(name, args.output.resolve(), passes) for name in args.models]
    write_json(args.output / "exports.json", {"version": VERSION, "models": reports})


if __name__ == "__main__":
    main()
