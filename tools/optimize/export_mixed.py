# SPDX-License-Identifier: MIT
"""Build M4 precision bundles before distribution; never run in a playback host."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import onnx
from onnx import TensorProto
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/export"))
from assets import build_reference, read_models, sha256, write_json
from export_reference import environment
from network import MixedOptimized


def audit(graph):
    """Check every convolution and sampling node using inferred ONNX types.

    Args:
        graph: Newly exported mixed graph with explicit Cast boundaries.
    """
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    types.update({v.name: v.data_type for v in inferred.graph.initializer})
    conv, sample = 0, 0
    for node in inferred.graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            assert all(types.get(name) == TensorProto.FLOAT16 for name in node.input if name), node.name
            assert types.get(node.output[0]) == TensorProto.FLOAT16, node.name
            conv += 1
        if node.op_type in ("GridSample", "Resize"):
            assert types.get(node.input[0]) == TensorProto.FLOAT, node.name
            assert types.get(node.output[0]) == TensorProto.FLOAT, node.name
            if node.op_type == "GridSample":
                assert types.get(node.input[1]) == TensorProto.FLOAT, node.name
                sample += 1
    assert conv > 0
    assert all(v.type.tensor_type.elem_type == TensorProto.FLOAT for v in [*graph.graph.input, *graph.graph.output])
    assert not any(n.op_type == "Clip" or n.domain for n in graph.graph.node)
    return {"conv_fp16": conv, "grid_sample_fp32": sample, "all_io": "fp32", "custom_operators": []}


def export(model_id, output):
    """Args:
        model_id: Pinned official model, using its own stage scales and channels.
        output: Project-local precision bundle root; M1/M3 assets stay untouched.
    """
    folder = output / model_id
    folder.mkdir(parents=True, exist_ok=True)
    base = ROOT / "build/optimized" / model_id
    descriptor = json.loads((base / "model.json").read_text(encoding="utf-8"))
    descriptor["variants"] = {}
    for name in ("full", "encoder", "synthesis"):
        source = base / f"{name}.onnx"
        target = folder / "fp32" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = descriptor["graph"] if name == "full" else descriptor["cache_graphs"][name]
        assert sha256(source) == expected["sha256"]
        shutil.copyfile(source, target)
        expected["file"] = f"fp32/{source.name}"
    for name in ("LICENSE.Practical-RIFE", "provenance.json"):
        shutil.copyfile(base / name, folder / name)
    reference, parameters = build_reference(read_models()[model_id])
    network = MixedOptimized(reference).eval().requires_grad_(False)
    exact_beta = 0
    for index, block in enumerate(network.synthesis.blocks):
        for actual, original in zip(block.convblock, getattr(reference, f"block{index}").convblock):
            assert actual.beta.dtype == torch.float32 and torch.equal(actual.beta, original.beta)
            exact_beta += 1
    a, b = torch.zeros(1, 3, 128, 256), torch.ones(1, 3, 128, 256)
    t = torch.full((1, 1, 1, 1), .5)
    assets = {}
    with torch.inference_mode():
        fa, fb = network.encode(a), network.encode(b)
        entries = {"full": (network, (a, b, t), ["image0", "image1", "timestep"], "output"),
                   "encoder": (network.encode, (a,), ["image"], "feature"),
                   "synthesis": (network.synthesis, (a, b, fa, fb, t),
                                 ["image0", "image1", "feature0", "feature1", "timestep"], "output")}
        for role, (module, inputs, names, result) in entries.items():
            path = folder / "fp16" / f"{role}.onnx"
            path.parent.mkdir(exist_ok=True)
            torch.onnx.export(module, inputs, str(path), input_names=names, output_names=[result],
                dynamic_axes={name: {2: "height", 3: "width"} for name in [*names, result] if name != "timestep"},
                opset_version=16, dynamo=False, do_constant_folding=True, external_data=False)
            graph = onnx.load(path)
            onnx.helper.set_model_props(graph, {"project": "RIFE-SuperBoost", "contract": "m4-mixed-v1",
                "role": role, "model_id": model_id, "network_precision": "fp16", "geometry": "fp32",
                "checkpoint_sha256": parameters["checkpoint_sha256"], "license": "MIT; Copyright (c) 2021 hzwer"})
            onnx.checker.check_model(graph, full_check=True)
            checks = audit(graph)
            onnx.save(graph, path)
            assets[role] = {"file": f"fp16/{path.name}", "sha256": sha256(path), "bytes": path.stat().st_size,
                            "audit": checks, "operators": dict(Counter(n.op_type for n in graph.graph.node))}
    descriptor["variants"]["fp16"] = {"precision": "fp16", "contract": "m4-mixed-v1",
        "graph": {k: assets["full"][k] for k in ("file", "sha256")},
        "cache_graphs": {role: {k: assets[role][k] for k in ("file", "sha256")} for role in ("encoder", "synthesis")}}
    write_json(folder / "model.json", descriptor)
    report = {"id": model_id, "assets": assets, "checkpoint_sha256": parameters["checkpoint_sha256"],
              "original_beta_parameters_exact_fp32": exact_beta,
              "descriptor_sha256": sha256(folder / "model.json"), "source_fp32_sha256": sha256(base / "full.onnx"),
              "policy": {"convolutions": "fp16", "residual_beta_accumulation": "fp32",
                         "flow_state_resize_grid_warp_blend": "fp32", "io_cache_storage": "fp32"},
              "environment": environment(), "exporter_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("network.py"))}}
    write_json(folder / "export.json", report)
    print(f"M4 precision bundle exported: {model_id}", flush=True)
    return report


def main():
    """Generate all requested precision bundles without loading TensorRT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(read_models()), choices=list(read_models()), help="Official model IDs.")
    parser.add_argument("--output", type=Path, default=ROOT / "build/m4-models", help="Development/release staging within this project.")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT): parser.error("Output must stay within this project")
    torch.set_num_threads(4)
    reports = [export(name, args.output.resolve()) for name in args.models]
    write_json(args.output / "exports.json", {"schema_version": 1, "models": reports})


if __name__ == "__main__":
    main()
