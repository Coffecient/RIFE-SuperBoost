# SPDX-License-Identifier: MIT
"""Rebuild development-only, unoptimized FP32 ONNX from official checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from assets import PROJECT, build_reference, read_models, sha256, verify_assets, write_json


def environment() -> dict[str, Any]:
    """Describe the exporter environment without loading a TensorRT runtime."""
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "onnx", "onnxruntime", "numpy")},
        "torch_cuda_build": torch.version.cuda,
    }


def audit_graph(path: Path, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Check generic operator semantics and every original parameter bit pattern.

    Args:
        path: Newly exported reference ONNX, without external tensor data.
        state: Strictly loaded official parameters, in their original channel order.
    """
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    if [(item.domain, item.version) for item in graph.opset_import] != [("", 16)]:
        raise ValueError("Reference graph must use only standard ONNX opset 16.")
    counts = Counter(node.op_type for node in graph.graph.node)
    if (counts["GridSample"] != 18 or counts["Clip"] or counts["ConvTranspose"] != 7 or
            counts["DepthToSpace"] != 5 or counts["Resize"] != 14):
        raise ValueError(f"Unexpected unoptimized reference structure: {counts}")
    expected_io = {"image0": [1, 3, "height", "width"], "image1": [1, 3, "height", "width"],
                   "timestep": [1, 1, 1, 1], "output": [1, 3, "height", "width"]}
    actual_io = {}
    for item in (*graph.graph.input, *graph.graph.output):
        if item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Non-FP32 reference IO: {item.name}")
        actual_io[item.name] = [dimension.dim_param or dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
    if actual_io != expected_io:
        raise ValueError(f"Unexpected reference IO contract: {actual_io}")
    for node in graph.graph.node:
        if node.domain:
            raise ValueError(f"Unexpected custom operator domain: {node.domain}")
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if node.op_type == "GridSample" and attributes != {"align_corners": 1, "mode": b"bilinear", "padding_mode": b"border"}:
            raise ValueError(f"Unexpected GridSample sampling rule: {attributes}")
        if node.op_type == "Resize" and (attributes.get("mode") != b"linear" or attributes.get("coordinate_transformation_mode") != b"half_pixel"):
            raise ValueError(f"Unexpected Resize sampling rule: {attributes}")
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    # The exporter may share identical beta tensors through Identity aliases.
    for node in graph.graph.node:
        if node.op_type == "Identity" and node.input[0] in tensors:
            tensors[node.output[0]] = tensors[node.input[0]]
    for name, value in state.items():
        if name not in tensors or not np.array_equal(tensors[name], value.cpu().numpy()):
            raise ValueError(f"ONNX parameter differs from the official checkpoint: {name}")
    for index in range(5):
        if tensors[f"block{index}.lastconv.0.weight"].shape[1] != 52:
            raise ValueError("A reference stage lost original output channels.")
    if any(item.data_type != onnx.TensorProto.FLOAT for item in graph.graph.initializer):
        raise ValueError("Reference parameters must all be float32.")
    return {"checker": "passed", "parameter_tensors_exact": len(state), "operator_counts": dict(sorted(counts.items())),
            "grid_sample": {"count": 18, "mode": "bilinear", "padding_mode": "border", "align_corners": True},
            "all_stage_deconvolution_outputs": 52, "custom_operators": [], "input_clamp": False,
            "input_output_shapes": actual_io}


def export_model(model: dict[str, Any], output: Path) -> dict[str, Any]:
    """Export one model with dynamic padded height/width and runtime scalar time.

    Args:
        model: Official model identity from models/manifest.json.
        output: Directory for generated ONNX and its export metadata.
    """
    print(f"[{model['id']}] Verifying assets and exporting FP32 reference", flush=True)
    inputs = verify_assets(model)
    network, parameter_report = build_reference(model)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"rife-{model['id']}-reference-fp32.onnx"
    temporary = destination.with_suffix(".onnx.tmp")
    sample0 = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    sample1 = torch.ones_like(sample0)
    timestep = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            network, (sample0, sample1, timestep), temporary,
            input_names=["image0", "image1", "timestep"], output_names=["output"],
            dynamic_axes={name: {2: "height", 3: "width"} for name in ("image0", "image1", "output")},
            opset_version=16, dynamo=False, do_constant_folding=False,
            keep_initializers_as_inputs=False, export_params=True, external_data=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
    contract = {
        "images": "float32 [1, 3, H, W], RGB plane order, matching shapes",
        "timestep": "float32 [1, 1, 1, 1], runtime scalar in [0, 1]",
        "output": "float32 [1, 3, H, W], padded output; caller crops to source dimensions",
        "spatial_axes": "dynamic H/W; aligned padded dimensions only, not arbitrary unpadded dimensions",
        "minimum_structural_alignment": model["reference_profile"]["padding"]["structural_alignment_candidate"],
        "official_preprocessing": "right/bottom zero padding to multiples of 128, then output crop",
        "scales": model["reference_profile"]["stage_scale_denominators"],
        "scale": 1, "additional_input_clamp": False, "project_optimization_passes": [],
    }
    graph = onnx.load(temporary)
    onnx.helper.set_model_props(graph, {
        "project": "RIFE-SuperBoost", "purpose": "unoptimized official FP32 reference",
        "model_id": model["id"], "checkpoint_sha256": model["checkpoint"]["sha256"],
        "license": "MIT; Copyright (c) 2021 hzwer; see bundled upstream notice",
        "contract": json.dumps(contract, sort_keys=True),
    })
    onnx.save(graph, temporary)
    audit = audit_graph(temporary, network.state_dict())
    temporary.replace(destination)
    report = {
        "schema_version": 1, "model_id": model["id"],
        "artifact": {"path": destination.relative_to(PROJECT).as_posix(), "bytes": destination.stat().st_size, "sha256": sha256(destination)},
        "contract": contract, "parameters": parameter_report, "graph_audit": audit,
        "verified_source_assets": inputs, "environment": environment(),
        "exporter": {"api": "torch.onnx.export", "dynamo": False, "opset": 16, "constant_folding": False,
                     "files": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}},
    }
    write_json(destination.with_suffix(".export.json"), report)
    print(f"[{model['id']}] Exported {destination.name}; all {len(network.state_dict())} parameters exact", flush=True)
    return report


def main() -> None:
    """Parse the development export command; never build engines or optimize graphs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *read_models()], default=["all"], help="Official model IDs to export; default: all four.")
    parser.add_argument("--output", type=Path, default=PROJECT / "build/reference", help="Generated reference directory under the project.")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("The output directory must stay within RIFE-SuperBoost.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = read_models()
    selected = list(models) if "all" in args.models else list(dict.fromkeys(args.models))
    reports = [export_model(models[name], output) for name in selected]
    write_json(output / "exports.json", {"schema_version": 1, "models": reports})


if __name__ == "__main__":
    main()
