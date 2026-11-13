# SPDX-License-Identifier: MIT
"""Validate reference exports against hash-verified original upstream inference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from assets import PROJECT, build_reference, read_models, sha256, verify_assets, write_json
from export_reference import audit_graph, environment
from reference import pad_pair, warp
from upstream_oracle import build_oracle

# These gates diagnose export regressions; they are not perceptual quality scores.
LIMITS = {"adapter_max_abs": 2e-6, "onnx_mean_abs": 1e-4, "onnx_rmse": 1e-3}


def difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """Measure finite, same-shape tensors without hiding outliers.

    Args:
        actual: Computed float output.
        expected: Corresponding reference output, with the same layout.
    """
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Output shape mismatch or NaN/Inf detected.")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "fraction_above_1_over_255": float(np.mean(delta > 1 / 255)), "shape": list(actual.shape)}


def make_pair(height: int, width: int, seed: int, extended_range: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reproducible moving color/texture/boundary probes, without media.

    Args:
        height: Unpadded source height.
        width: Unpadded source width.
        seed: Local RNG seed, independent of test ordering.
        extended_range: Include negative and above-one values to detect clamping.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    red = x / max(1, width - 1)
    green = y / max(1, height - 1)
    blue = ((x // 7 + y // 11) % 2).astype(np.float32)
    texture = rng.random((3, height, width), dtype=np.float32)
    image0 = 0.85 * np.stack((red, green, blue)) + 0.15 * texture
    image0[:, :max(1, height // 12), -max(1, width // 9):] = np.array([1.0, 0.0, 0.25])[:, None, None]
    image1 = np.roll(image0, shift=(max(1, height // 17), -max(1, width // 13)), axis=(1, 2)).copy()
    if extended_range:
        image0 = image0 * 1.5 - 0.25
        image1 = image1 * 1.5 - 0.25
    return torch.from_numpy(image0[None].copy()), torch.from_numpy(image1[None].copy())


def warp_checks(oracle: Any) -> list[dict[str, Any]]:
    """Check displacement direction, boundaries, subpixels, and non-square grids.

    Args:
        oracle: Original warp implementation loaded from a verified snapshot.
    """
    image = torch.arange(3 * 7 * 11, dtype=torch.float32).reshape(1, 3, 7, 11) / 230
    results = []
    for dx, dy in ((0.0, 0.0), (1.0, 0.0), (-0.5, 1.25), (100.0, -100.0)):
        flow = torch.zeros(1, 2, 7, 11)
        flow[:, 0] = dx
        flow[:, 1] = dy
        expected = oracle.warp(image, flow)
        actual = warp(image, flow)
        stats = difference(actual.numpy(), expected.numpy())
        if stats["max_abs"] != 0:
            raise AssertionError(f"Warp differs from official sampling: {stats}")
        if dx == 1.0 and dy == 0.0:
            torch.testing.assert_close(actual[..., :-1], image[..., 1:], rtol=0, atol=1e-6)
            torch.testing.assert_close(actual[..., -1], image[..., -1], rtol=0, atol=1e-6)
        if dx == 100.0:
            torch.testing.assert_close(actual, image[:, :, :1, -1:].expand_as(actual), rtol=0, atol=1e-6)
        results.append({"dx": dx, "dy": dy, "official_comparison": stats})
    return results


def compare_case(network: Any, oracle: Any, session: ort.InferenceSession,
                 height: int, width: int, time: float, alignment: int,
                 extended_range: bool = False) -> dict[str, Any]:
    """Compare original PyTorch, reference PyTorch, and CPU ONNX at one input.

    Args:
        network: Strictly loaded unoptimized reference adapter.
        oracle: Original official inference method and network.
        session: ONNX Runtime CPU session with graph optimization disabled.
        height: Unpadded source height.
        width: Unpadded source width.
        time: Runtime interpolation scalar, not an export-time constant.
        alignment: Padding alignment under test, either 128 or the candidate.
        extended_range: Probe the absence of an extra input/output clamp.
    """
    image0, image1 = make_pair(height, width, 20261111, extended_range)
    left, right = pad_pair(image0, image1, alignment)
    expected_shape = (1, 3, height + (-height) % alignment, width + (-width) % alignment)
    if tuple(left.shape) != expected_shape:
        raise AssertionError("Incorrect padding dimensions.")
    torch.testing.assert_close(left[..., :height, :width], image0, rtol=0, atol=0)
    if height < left.shape[-2] and torch.count_nonzero(left[..., height:, :]):
        raise AssertionError("Bottom padding is not zero.")
    if width < left.shape[-1] and torch.count_nonzero(left[..., :, width:]):
        raise AssertionError("Right padding is not zero.")
    timestep = torch.full((1, 1, 1, 1), time, dtype=torch.float32)
    official = oracle.inference(left, right, timestep=time, scale=1.0)
    adapter = network(left, right, timestep)
    adapter_stats = difference(adapter.numpy(), official.numpy())
    if adapter_stats["max_abs"] > LIMITS["adapter_max_abs"]:
        raise AssertionError(f"Reference adapter changed official inference: {adapter_stats}")
    actual = session.run(["output"], {"image0": left.numpy(), "image1": right.numpy(), "timestep": timestep.numpy()})[0]
    stats = difference(actual, official.numpy())
    cropped = difference(actual[..., :height, :width], official.numpy()[..., :height, :width])
    if any(item["mean_abs"] > LIMITS["onnx_mean_abs"] or item["rmse"] > LIMITS["onnx_rmse"]
           for item in (stats, cropped)):
        raise AssertionError(f"ONNX export exceeded numerical gates at {width}x{height}, t={time}, alignment={alignment}: {stats}")
    return {"height": height, "width": width, "timestep": time, "alignment": alignment,
            "extended_input_range": extended_range, "adapter_vs_official": adapter_stats,
            "onnx_vs_official": stats, "cropped_onnx_vs_official": cropped,
            "output_min": float(actual.min()), "output_max": float(actual.max())}


def alignment_checks(network: Any, candidate: int) -> list[dict[str, Any]]:
    """Record padding differences on synthetic probes without claiming quality.

    Args:
        network: Reference adapter, without any project graph optimizations.
        candidate: Model-specific structural alignment candidate.
    """
    results = []
    for height, width in ((65, 193), (191, 321)):
        left, right = make_pair(height, width, 425)
        timestep = torch.full((1, 1, 1, 1), 0.5)
        base = network(*pad_pair(left, right, 128), timestep)[..., :height, :width].numpy()
        trial = network(*pad_pair(left, right, candidate), timestep)[..., :height, :width].numpy()
        border = np.zeros((height, width), dtype=bool)
        border[-16:, :] = True
        border[:, -16:] = True
        results.append({"height": height, "width": width, "candidate_alignment": candidate,
                        "candidate_vs_128": difference(trial, base),
                        "right_bottom_16px": difference(trial[..., border], base[..., border]),
                        "interpretation": "Synthetic diagnostic only; release alignment awaits real-content quality review."})
    return results


def large_shape_checks(model: dict[str, Any], device: str) -> dict[str, Any]:
    """Exercise full-resolution official/reference FP32 inference on a device.

    Args:
        model: Model entry defining the original network and adapter.
        device: Explicit device for this optional resource-intensive check.
    """
    network, _ = build_reference(model)
    network.to(device)
    oracle = build_oracle(model, device)
    results = []
    for height, width in ((720, 1280), (1080, 1920), (1440, 2560), (2160, 3840), (479, 853), (1280, 720)):
        print(f"[{model['id']}] {device} shape check {width}x{height}", flush=True)
        source0, source1 = make_pair(height, width, 426)
        left, right = pad_pair(source0.to(device), source1.to(device), 128)
        timestep = torch.full((1, 1, 1, 1), 0.375, device=device)
        actual = network(left, right, timestep)[..., :height, :width].cpu().numpy()
        official = oracle.inference(left, right, timestep=0.375, scale=1.0)[..., :height, :width].cpu().numpy()
        stats = difference(actual, official)
        if stats["max_abs"] > LIMITS["adapter_max_abs"]:
            raise AssertionError(f"Full-size adapter differs from official inference: {stats}")
        results.append({"height": height, "width": width, "alignment": 128,
                        "adapter_vs_official": stats, "output_min": float(actual.min()), "output_max": float(actual.max())})
        oracle.grid_cache.clear()
        del left, right, actual, official
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {"device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
            "tf32": False, "deterministic_algorithms": True, "cases": results,
            "scope": "PyTorch official/adapter only; no TensorRT engine or playback certification."}


def verify_model(model: dict[str, Any], directory: Path, full_size_device: str | None) -> dict[str, Any]:
    """Verify source provenance, graph integrity, scalar time, padding, and outputs.

    Args:
        model: One pinned official model entry.
        directory: Directory containing the reference ONNX and export metadata.
        full_size_device: Optional explicit device for 720p through native-4K checks.
    """
    print(f"[{model['id']}] Comparing official, adapter, and ONNX Runtime CPU", flush=True)
    verify_assets(model)
    path = directory / f"rife-{model['id']}-reference-fp32.onnx"
    export_report = json.loads(path.with_suffix(".export.json").read_text(encoding="utf-8"))
    if sha256(path) != export_report["artifact"]["sha256"]:
        raise ValueError("The exported graph digest does not match its export report.")
    network, parameters = build_reference(model)
    audit = audit_graph(path, network.state_dict())
    oracle = build_oracle(model)
    observed_scales = []

    def capture_scale(module, args, kwargs):
        """Record the scale actually selected by the original inference method.

        Args:
            module: Official stage invoking the hook.
            args: Positional stage inputs.
            kwargs: Keyword inputs including the original scale denominator.
        """
        observed_scales.append(kwargs["scale"])

    handles = [getattr(oracle.flownet, f"block{index}").register_forward_pre_hook(capture_scale, with_kwargs=True)
               for index in range(5)]
    probe0, probe1 = make_pair(128, 128, 425)
    oracle.inference(probe0, probe1, timestep=0.5, scale=1.0)
    for handle in handles:
        handle.remove()
    if observed_scales != model["reference_profile"]["stage_scale_denominators"]:
        raise AssertionError(f"Manifest scales differ from the official entry: {observed_scales}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    cases = []
    # Identical dimensions at several times catch accidentally frozen time inputs.
    probes = [(128, 128, 0.5, False), (129, 193, 0.0, False), (129, 193, 0.25, False),
              (129, 193, 0.5, False), (129, 193, 0.75, False), (129, 193, 1.0, False),
              (33, 65, 1 / 3, True), (1, 1, 0.5, False), (191, 257, 0.375, False),
              (479, 853, 0.375, False), (720, 1280, 0.375, False)]
    for height, width, time, extended in probes:
        cases.append(compare_case(network, oracle, session, height, width, time, 128, extended))
    candidate = model["reference_profile"]["padding"]["structural_alignment_candidate"]
    for height, width in ((64, 192), (129, 193)):
        cases.append(compare_case(network, oracle, session, height, width, 0.5, candidate))
    pair = pad_pair(*make_pair(129, 193, 20261111))
    feeds = {"image0": pair[0].numpy(), "image1": pair[1].numpy()}
    timed = [session.run(None, {**feeds, "timestep": np.full((1, 1, 1, 1), time, np.float32)})[0] for time in (0.25, 0.75)]
    time_difference = difference(timed[0], timed[1])
    if time_difference["mean_abs"] < 1e-5:
        raise AssertionError("The runtime timestep appears frozen.")
    report = {"model_id": model["id"], "status": "passed", "artifact": export_report["artifact"],
              "parameters": parameters, "official_observed_scales": observed_scales,
              "graph_audit": audit, "warp_probes": warp_checks(oracle),
              "cpu_numerical_cases": cases, "runtime_time_response": time_difference,
              "alignment_diagnostics": alignment_checks(network, candidate),
              "release_alignment_status": "not_frozen_pending_real_content_review"}
    del session, oracle, network
    gc.collect()
    if full_size_device:
        report["full_resolution"] = large_shape_checks(model, full_size_device)
    print(f"[{model['id']}] All reference checks passed", flush=True)
    return report


def main() -> None:
    """Run development checks; large-shape GPU tests must be requested explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *read_models()], default=["all"], help="Official model IDs to verify; default: all four.")
    parser.add_argument("--directory", type=Path, default=PROJECT / "build/reference", help="Directory containing reference ONNX exports.")
    parser.add_argument("--full-size-device", help="Also compare official/adapter at 720p, 1080p, 1440p, 4K, odd and portrait sizes; e.g. cuda:0.")
    parser.add_argument("--report", type=Path, default=PROJECT / "docs/evidence/m1-reference-validation.json", help="Validation JSON destination inside the project.")
    args = parser.parse_args()
    if not args.report.resolve().is_relative_to(PROJECT):
        parser.error("The report destination must stay within RIFE-SuperBoost.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    models = read_models()
    selected = list(models) if "all" in args.models else list(dict.fromkeys(args.models))
    # Write progress atomically, so interrupted runs never claim all models passed.
    report = {"schema_version": 1, "milestone": "M1", "status": "running", "limits": LIMITS,
              "gate_rationale": "Cross-framework FP32 accumulation and dynamic linspace/GridSample rounding are amplified by five flow stages, especially on sharp synthetic textures. Exact parameter/semantic checks are mandatory; MAE/RMSE gates apply to padded and cropped outputs. Maximum error and outlier fractions remain reported diagnostics, not bit-identity gates. Layer diagnostics are retained separately.",
              "environment": environment(), "onnx_provider": "CPUExecutionProvider", "onnx_graph_optimization": "disabled",
              "source_comparison": "Original pinned IFNet and inference method, original warp; only package import binding/device selection adapted.",
              "models": [], "limitations": ["Synthetic numerical probes are not a hard-case or perceptual quality evaluation.",
              "Full-resolution device probes exercise PyTorch, not ONNX Runtime or TensorRT at those resolutions.",
              "Release alignment and backend support remain M3/M5/M6 decisions."]}
    write_json(args.report, report)
    try:
        with torch.inference_mode():
            for name in selected:
                report["models"].append(verify_model(models[name], args.directory, args.full_size_device))
                write_json(args.report, report)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.report, report)


if __name__ == "__main__":
    main()
