# SPDX-License-Identifier: MIT
"""Pinned asset verification and strict checkpoint loading for development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from reference import ReferenceNet

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "models"
MANIFEST = MODELS / "manifest.json"


def sha256(path: Path) -> str:
    """Return a file digest.

    Args:
        path: File to read without changing its contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_models() -> dict[str, dict[str, Any]]:
    """Read the checked-in model identities; no network access is performed."""
    return {item["id"]: item for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]}


def verify_file(record: dict[str, Any]) -> Path:
    """Resolve an asset under models/ and require its pinned size and digest.

    Args:
        record: Manifest entry with path, bytes, and sha256 fields.
    """
    path = (MODELS / record["path"]).resolve()
    if not path.is_relative_to(MODELS.resolve()):
        raise ValueError(f"Asset path escapes models/: {record['path']}")
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Asset digest or size mismatch: {record['path']}")
    return path


def verify_assets(model: dict[str, Any]) -> dict[str, str]:
    """Verify model inputs and shared upstream license/source snapshots.

    Args:
        model: One model identity from the checked-in manifest.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [model["archive"], model["checkpoint"], *model["source_files"], *manifest["upstream_snapshots"]]
    return {record["path"]: sha256(verify_file(record)) for record in records}


def load_checkpoint(model: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load tensors and separate an exact schema of unused training parameters.

    Args:
        model: Verified official model identity. Every excluded key is reported.
    """
    path = verify_file(model["checkpoint"])
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("The checkpoint must be a nonempty tensor state dictionary.")
    state = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("The checkpoint contains a non-tensor or non-string entry.")
        key = name.removeprefix("module.")
        if key in state:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite float32 parameter: {key}")
        state[key] = value
    training_shapes = training_only_shapes(model["reference_profile"]["encoding_channels"])
    training_keys = {key for key in state if key.startswith(("teacher.", "caltime."))}
    if training_keys != training_shapes.keys():
        raise ValueError("The checkpoint training-only key set differs from the audited schema.")
    ignored = []
    for key, shape in training_shapes.items():
        value = state.pop(key)
        if tuple(value.shape) != shape:
            raise ValueError(f"Unexpected training-only parameter shape: {key}")
        ignored.append({"key": key, "shape": list(shape), "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()})
    return state, {
        "checkpoint_sha256": model["checkpoint"]["sha256"],
        "raw_tensor_count": len(raw),
        "tensor_count": len(state),
        "parameter_elements": sum(value.numel() for value in state.values()),
        "stripped_ddp_prefix_count": sum(key.startswith("module.") for key in raw),
        "ignored_training_only": ignored,
        "training_schema_note": "Exact teacher/caltime keys and shapes audited from pinned checkpoints; the upstream inactive caltime comment has stale shapes.",
        "strict": True,
    }


def training_only_shapes(encoding_channels: int) -> dict[str, tuple[int, ...]]:
    """Define all forty audited training-only keys; no prefix wildcard is allowed.

    Args:
        encoding_channels: Four or sixteen, selecting the known checkpoint schema.
    """
    if encoding_channels not in (4, 16):
        raise ValueError("Unknown training-only checkpoint schema.")
    shapes = {
        "teacher.conv0.0.0.weight": (32, 23 + 2 * encoding_channels, 3, 3),
        "teacher.conv0.0.0.bias": (32,),
        "teacher.conv0.1.0.weight": (64, 32, 3, 3),
        "teacher.conv0.1.0.bias": (64,),
        "teacher.lastconv.0.weight": (64, 52, 4, 4),
        "teacher.lastconv.0.bias": (52,),
    }
    for index in range(8):
        prefix = f"teacher.convblock.{index}"
        shapes.update({f"{prefix}.beta": (1, 64, 1, 1), f"{prefix}.conv.weight": (64, 64, 3, 3),
                       f"{prefix}.conv.bias": (64,)})
    # These are the actual pinned checkpoint shapes, not the stale source comment.
    for index, input_channels, output_channels in ((0, 9 + 2 * encoding_channels, 32), (2, 32, 64),
                                                   (4, 64, 64), (6, 64, 64), (8, 64, 1)):
        shapes[f"caltime.{index}.weight"] = (output_channels, input_channels, 3, 3)
        shapes[f"caltime.{index}.bias"] = (output_channels,)
    return shapes


def build_reference(model: dict[str, Any]) -> tuple[ReferenceNet, dict[str, Any]]:
    """Build a CPU FP32 reference with exact key, shape, and value matching.

    Args:
        model: Manifest entry selecting architecture and official stage scales.
    """
    profile = model["reference_profile"]
    network = ReferenceNet(profile["encoding_channels"], 24 if model["id"] == "4.25-lite" else 32,
                           profile["stage_scale_denominators"])
    state, report = load_checkpoint(model)
    network.load_state_dict(state, strict=True)
    for name, value in network.state_dict().items():
        if not torch.equal(value, state[name]):
            raise ValueError(f"Parameter changed while loading: {name}")
    return network.eval().requires_grad_(False), report


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON report, refusing NaN and infinity.

    Args:
        path: Destination report path.
        value: JSON-compatible report content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
