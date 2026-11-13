# SPDX-License-Identifier: MIT
"""Export prepared dynamic TensorRT fusions from the audited official weights."""

import argparse
import json
from pathlib import Path
import sys

import onnx
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/export"))
from assets import build_reference, read_models, sha256, write_json
from network import Optimized, MixedOptimized
from tensorrt_network import FusedEncoder, FusedSynthesis


def export(model_id, output):
    """Args:
        model_id: Official model with a strictly checked checkpoint and profile.
        output: Project-local development asset directory, never a player cache.
    """
    reference, parameters = build_reference(read_models()[model_id])
    folder = output / model_id
    folder.mkdir(parents=True, exist_ok=True)
    result = {"model_id": model_id, "checkpoint_sha256": parameters["checkpoint_sha256"],
              "scales": list(reference.scales), "alignment": 128, "padding": "zero", "variants": {}}
    torch.manual_seed(20261112)
    for precision in ("fp32", "fp16"):
        oracle = Optimized(reference) if precision == "fp32" else MixedOptimized(reference)
        for storage in ("fp32", "fp16"):
            name = f"trt-fused-{precision}-io-{storage}"
            encoder = FusedEncoder(reference, precision).eval().requires_grad_(False)
            synthesis = FusedSynthesis(reference, precision, storage).eval().requires_grad_(False)
            dtype = torch.float16 if storage == "fp16" else torch.float32
            a, b = [torch.rand(1, 3, 128, 256).to(dtype) for _ in range(2)]
            t = torch.full((1, 1, 1, 1), .5)
            with torch.inference_mode():
                fa, fb = encoder(a), encoder(b)
                actual = synthesis(a, b, fa, fb, t)
                expected = oracle(a.float(), b.float(), t).to(dtype)
                delta = (actual.float()-expected.float()).abs()
                assert delta.max() < (1e-3 if storage == "fp16" else 1e-4), (model_id, name, delta.max())
                entry = {"precision": precision, "io_precision": storage, "feature_precision": precision,
                         "execution": "trt-fused-v1", "contract": "m3-trt-fused-v1", "cache_graphs": {},
                         "eager_max_abs_vs_portable": float(delta.max())}
                for role, module, inputs, names, out in (
                    ("encoder", encoder, (a,), ["image"], "feature"),
                    ("synthesis", synthesis, (a, b, fa, fb, t), ["image0", "image1", "feature0", "feature1", "timestep"], "output")):
                    path = folder / f"{name}-{role}.onnx"
                    torch.onnx.export(module, inputs, str(path), input_names=names, output_names=[out],
                        dynamic_axes={n: {2: "height", 3: "width"} for n in [*names, out] if n != "timestep"},
                        opset_version=16, custom_opsets={"rsb.trt": 1}, dynamo=False, do_constant_folding=True)
                    graph = onnx.load(path)
                    onnx.helper.set_model_props(graph, {"project": "RIFE-SuperBoost", "model_id": model_id,
                        "contract": "m3-trt-fused-v1", "role": role, "precision": precision, "storage": storage,
                        "geometry": "fp32", "checkpoint_sha256": parameters["checkpoint_sha256"],
                        "scales": json.dumps(list(reference.scales)), "license": "MIT; Copyright (c) 2021 hzwer"})
                    onnx.checker.check_model(graph)
                    nodes = [n.op_type for n in graph.graph.node if n.domain == "rsb.trt"]
                    assert sorted(nodes) == ([] if role == "encoder" else sorted(["RsbStagePack", *(["RsbFlowWarpPack"]*4), "RsbWarpBlend"]))
                    onnx.save(graph, path)
                    entry["cache_graphs"][role] = {"file": path.name, "sha256": sha256(path)}
                result["variants"][name] = entry
            print(f"Exported {model_id}/{name}", flush=True)
    result["generator_sha256"] = {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("tensorrt_network.py"))}
    write_json(folder / "variants.json", result)


def main():
    """Export floating layouts only; independent PTQ/QAT weights stay separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(read_models()), default=list(read_models()), help="Audited official model IDs.")
    parser.add_argument("--output", type=Path, default=ROOT / "build/m3-fused", help="Project-local prepared assets.")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must stay inside the project")
    torch.set_num_threads(4)
    for model_id in args.models:
        export(model_id, args.output)


if __name__ == "__main__":
    main()
