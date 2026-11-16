# SPDX-License-Identifier: MIT
"""Publish split precision graphs sharing FP32 base weights before distribution."""

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.external_data_helper import set_external_data
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"4.25": "425", "4.25-lite": "425_lite", "4.26": "426", "4.26-large": "426_large",
          "4.25-scale8": "425_scale8", "4.26-scale8": "426_scale8"}
RELEASE_VERSION = tomllib.loads((ROOT / "packages/core/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
CAST_PREFIX = "rsb_shared_weight_cast_"


def share_float_weights(graph: onnx.ModelProto, base: onnx.ModelProto) -> int:
    """Replace rounded FP16 parameters with explicit Casts of matching FP32 bases.

    Args:
        graph: Mutable mixed graph; its computation types remain explicit.
        base: Matching optimized FP32 graph with identical parameter layouts.
    """
    originals = {t.name: t for t in base.graph.initializer}
    names = {v for n in graph.graph.node for v in (*n.input, *n.output)}
    names.update(t.name for t in graph.graph.initializer)
    casts = []
    for tensor in graph.graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT16:
            continue
        # M4's encoder wrapper adds network.; synthesis parameter names already match.
        original = originals[tensor.name if tensor.name in originals else tensor.name.removeprefix("network.")]
        if original.data_type != onnx.TensorProto.FLOAT or original.dims != tensor.dims:
            raise ValueError(f"Shared base type/shape mismatch: {tensor.name}")
        if numpy_helper.to_array(original).astype(np.float16).tobytes() != tensor.raw_data:
            raise ValueError(f"Independent weights require a separate PTQ/QAT variant: {tensor.name}")
        output, base_name = tensor.name, tensor.name + ".rsb_base_fp32"
        if base_name in names:
            raise ValueError(f"Shared weight name collision: {base_name}")
        names.add(base_name)
        tensor.name, tensor.data_type, tensor.raw_data = base_name, onnx.TensorProto.FLOAT, original.raw_data
        casts.append(helper.make_node("Cast", [base_name], [output], to=onnx.TensorProto.FLOAT16,
                                      name=CAST_PREFIX + str(len(casts))))
    nodes = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend([*casts, *nodes])
    return len(casts)


def fold_weight_casts(graph: onnx.ModelProto) -> onnx.ModelProto:
    """Reconstruct the old graph for an exact computation/parameter comparison.

    Args:
        graph: Loaded graph, including all standard ONNX external data.
    """
    result = deepcopy(graph)
    casts = {n.input[0]: n.output[0] for n in result.graph.node if n.name.startswith(CAST_PREFIX)}
    for tensor in result.graph.initializer:
        if tensor.name in casts:
            raw = numpy_helper.to_array(tensor).astype(np.float16).tobytes()
            tensor.name, tensor.data_type, tensor.raw_data = casts[tensor.name], onnx.TensorProto.FLOAT16, raw
        if tensor.data_location == onnx.TensorProto.DEFAULT:
            tensor.ClearField("data_location")
    nodes = [n for n in result.graph.node if not n.name.startswith(CAST_PREFIX)]
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return result


def digest(path: Path) -> str:
    """Args:
        path: Prepared source or published model asset.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Args:
        path: Project-local metadata destination.
        value: Provenance or validation data, containing no non-finite numbers.
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(model_id: str, suffix: str, upstream: dict) -> dict:
    """Build shared floating assets; independent PTQ/QAT weights use another variant.

    Args:
        model_id: Official model ID with a validated M4 precision bundle.
        suffix: Python import suffix for the independent model distribution.
        upstream: Pinned original source, checkpoint and license metadata.
    """
    source = ROOT / "build/m4-models" / model_id
    target = ROOT / f"packages/models/{model_id}/src/rife_superboost_model_{suffix}/assets"
    target.mkdir(parents=True, exist_ok=True)
    descriptor = json.loads((source / "model.json").read_text(encoding="utf-8"))
    pool, graphs = {}, []
    originals = {}
    entries = [(precision, descriptor if precision == "fp32" else descriptor["variants"]["fp16"], source)
               for precision in ("fp32", "fp16")]
    fused_folder = ROOT / "build/m3-fused" / model_id
    fused_record = json.loads((fused_folder / "variants.json").read_text(encoding="utf-8"))
    if fused_record["checkpoint_sha256"] != upstream["checkpoint"]["sha256"]:
        raise ValueError("Fusion assets must derive from the exact official checkpoint")
    if fused_record["scales"] != descriptor["scales"] or fused_record["model_id"] != model_id:
        raise ValueError("Portable and TensorRT graphs must use the same inference profile")
    descriptor["backend_variants"] = deepcopy(fused_record["variants"])
    entries.extend((entry["precision"], entry, fused_folder) for entry in descriptor["backend_variants"].values())
    weight_path = target / "weights.bin"
    with weight_path.open("wb") as weights:
        for precision, entry, source_folder in entries:
            # Full graphs remain in the M3/M4 development assets.
            entry.pop("graph", None)
            for role in ("encoder", "synthesis"):
                asset = entry["cache_graphs"][role]
                original = source_folder / asset["file"]
                if digest(original) != asset["sha256"]:
                    raise ValueError(f"Prepared source graph changed: {original}")
                graph = onnx.load(original)
                casts = share_float_weights(graph, onnx.load(source / "fp32" / (role + ".onnx"))) if precision == "fp16" else 0
                externalized = 0
                for tensor in graph.graph.initializer:
                    if len(tensor.raw_data) < 1024:
                        continue
                    if tensor.data_type != onnx.TensorProto.FLOAT:
                        raise ValueError(f"Unexpected external base weight type: {tensor.name}")
                    raw = tensor.raw_data
                    key = hashlib.sha256(raw).hexdigest()
                    if key not in pool:
                        # Stable offsets are shared across precisions and graph roles.
                        weights.write(b"\0" * ((-weights.tell()) % 64))
                        pool[key] = (weights.tell(), len(raw))
                        weights.write(raw)
                    offset, length = pool[key]
                    assert length == len(raw)
                    set_external_data(tensor, "weights.bin", offset=offset, length=length)
                    tensor.ClearField("raw_data")
                    externalized += 1
                source_file = asset["file"]
                asset["file"] = original.name if source_folder == fused_folder else f"{precision}-{role}.onnx"
                output = target / asset["file"]
                output.parent.mkdir(exist_ok=True)
                output.write_bytes(graph.SerializeToString())
                originals[asset["file"]] = original
                graphs.append({"precision": precision, "role": role, "file": asset["file"], "source_file": source_file,
                               "execution": entry.get("execution", "portable"), "io_precision": entry.get("io_precision", "fp32"),
                               "source_sha256": asset["sha256"], "sha256": digest(output),
                               "source_bytes": original.stat().st_size, "bytes": output.stat().st_size,
                               "externalized_initializers": externalized, "shared_fp32_weight_casts": casts})
                asset["sha256"] = graphs[-1]["sha256"]
    descriptor.update(schema_version=2, variant_id="float", weight_family="base-fp32", quantization="none",
                      validated_backends=["tensorrt"], layout="split-shared-float-v2", distribution_revision="m3-completion-v1")
    descriptor["external_data"] = [{"file": "weights.bin", "sha256": digest(weight_path), "bytes": weight_path.stat().st_size}]
    descriptor["distribution_version"] = RELEASE_VERSION
    descriptor["provenance_file"] = "provenance.json"
    write_json(target / "model.json", descriptor)
    shutil.copyfile(ROOT / "models/licenses/practical-rife/LICENSE", target / "LICENSE.Practical-RIFE")
    provenance = {"schema_version": 2, "id": model_id, "variant_id": "float", "upstream": upstream,
                  "prepared_source_descriptor_sha256": digest(source / "model.json"),
                  "packaging": "Split only; shared FP32 external weights; explicit constant Casts in prepared FP16 graphs",
                  "graphs": graphs, "external_data": descriptor["external_data"],
                  "generator_sha256": digest(Path(__file__)), "license": "MIT; Copyright (c) 2021 hzwer"}
    if "inference_profile" in descriptor:
        provenance["inference_profile"] = descriptor["inference_profile"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    # Fold constant weight Casts before ORT's CPU FP16 emulation inserts FP32 casts.
    # Disabling folding lets that emulation bypass FP16 rounding of constant weights.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    rng = np.random.default_rng(20261111)
    images = [rng.uniform(.1, .9, (1, 3, 128, 256)).astype(np.float32) for _ in range(2)]
    for precision in ("fp32", "fp16"):
        encoder = ort.InferenceSession(str(source / precision / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
        features = [encoder.run(None, {"image": image})[0] for image in images]
        for record in (g for g in graphs if g["precision"] == precision):
            original, published = originals[record["file"]], target / record["file"]
            # Folding only the added constant Casts must recover every previous field.
            a, b = fold_weight_casts(onnx.load(original)), fold_weight_casts(onnx.load(published))
            assert a == b, (model_id, record["file"], "protobuf mismatch")
            onnx.checker.check_model(str(published), full_check=True)
            record["all_fields_and_tensor_bytes_exact_after_constant_cast_folding"] = True
            if record["execution"] == "trt-fused-v1":
                record["native_validation_required"] = True
                continue
            feeds = {"image": images[0]} if record["role"] == "encoder" else {
                "image0": images[0], "image1": images[1], "timestep": np.full((1, 1, 1, 1), 1/3, np.float32)}
            if record["role"] == "synthesis":
                feeds.update(feature0=features[0], feature1=features[1])
            expected = ort.InferenceSession(str(original), options, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
            actual = ort.InferenceSession(str(published), options, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
            np.testing.assert_array_equal(actual, expected)
            record["all_fields_and_tensor_bytes_exact_after_constant_cast_folding"] = True
            record["ort_output_bit_exact"] = True
            record["ort_optimization_level"] = "basic-constant-folding-before-cpu-fp16-emulation"
    # Delete only obsolete generated full graphs after all new graphs have passed.
    for stale in target.glob("*-full.onnx"):
        stale.unlink()
    for precision in ("fp32", "fp16"):
        obsolete = target / precision
        if obsolete.is_dir():
            for role in ("full", "encoder", "synthesis"):
                (obsolete / (role + ".onnx")).unlink(missing_ok=True)
            obsolete.rmdir()
    write_json(target / "provenance.json", provenance)
    expected_files = {"weights.bin", "model.json", "provenance.json", "LICENSE.Practical-RIFE", *(g["file"] for g in graphs)}
    if {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} != expected_files:
        raise ValueError(f"Unexpected model assets: {target}")
    before = sum(g["source_bytes"] for g in graphs)
    after = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"Prepared {model_id}: {before:,} -> {after:,} bytes; {len(graphs)} split graphs prepared", flush=True)
    return {"id": model_id, "variant_id": "float", "layout": descriptor["layout"], "source_graph_bytes": before, "asset_bytes": after,
            "descriptor_sha256": digest(target / "model.json"), "graphs": graphs,
            "external_data": descriptor["external_data"], "provenance_sha256": digest(target / "provenance.json")}


def main() -> None:
    """Prepare all four independent wheels; no playback code imports this tool."""
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    upstream = {m["id"]: m for m in manifest["models"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS), help="Prepared official models to package.")
    args = parser.parse_args()
    profiles = {p["id"]: p for p in json.loads((ROOT / "models/inference-profiles.json").read_text(encoding="utf-8"))["profiles"]}
    records = [prepare(model_id, MODELS[model_id], upstream[profiles[model_id]["base_model"] if model_id in profiles else model_id])
               for model_id in args.models]
    manifest_path = ROOT / "models/distribution-manifest.json"
    existing = {r["id"]: r for r in json.loads(manifest_path.read_text(encoding="utf-8"))["models"]} if manifest_path.exists() else {}
    existing.update({r["id"]: r for r in records})
    write_json(manifest_path, {"schema_version": 2, "version": RELEASE_VERSION, "revision": "m3-completion-v1",
                              "models": [existing[key] for key in MODELS if key in existing]})


if __name__ == "__main__":
    main()
