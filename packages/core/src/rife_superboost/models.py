# SPDX-License-Identifier: MIT
"""Explicit model descriptors and optional installed-model discovery."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import entry_points
import json
from pathlib import Path
import re

from .errors import ModelNotInstalledError, RifeError


def file_digest(path: Path) -> str:
    """Compute a graph or binary identity without loading the whole file.

    Args:
        path: Existing file to hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


@dataclass(frozen=True)
class Model:
    """Read-only model identity consumed by optional backends.

    Args:
        model_id: Official or installed model identity.
        graph: Optional full development graph; user packages carry split graphs.
        sha256: Expected full graph digest, or None when no full graph is supplied.
        alignment: Required right/bottom zero-padding alignment.
        purpose: Reference or production marker from the model descriptor.
        encoder: Optional pre-exported single-frame encoder graph.
        encoder_sha256: Expected encoder digest.
        synthesis: Optional pre-exported synthesis graph consuming cached features.
        synthesis_sha256: Expected synthesis digest.
        encoding_channels: Four or sixteen for a split model; zero for full-only.
        contract: Versioned prepared-graph contract, included in engine identity.
        precision: Prepared neural arithmetic policy, separate from RGB storage.
        fp16: Optional pre-generated mixed variant from the same model bundle.
        external_data: Prepared ONNX external weight files and their SHA-256 digests.
            Shared across graphs; verified without importing ONNX or changing files.
        variant_id: Independent weight variant identity, such as float or a QAT ID.
        weight_family: Identity of the shared base or independent trained weights.
        quantization: none, ptq or qat; backends must explicitly support the policy.
        validated_backends: Backends with evidence for this prepared graph family.
        backend_variants: Prepared backend-specific layouts sharing the base weights.
        execution: portable or trt-fused-v1; no graph conversion occurs on the user host.
        io_precision: RGB graph input/output storage, independent of neural precision.
        feature_precision: Actual encoder output and feature-cache storage.
    """

    model_id: str
    graph: Path | None = None
    sha256: str | None = None
    alignment: int = 128
    purpose: str = "reference"
    encoder: Path | None = None
    encoder_sha256: str | None = None
    synthesis: Path | None = None
    synthesis_sha256: str | None = None
    encoding_channels: int = 0
    contract: str = "rgb-fp32-two-images-runtime-scalar-v1"
    precision: str = "fp32"
    fp16: Model | None = None
    external_data: tuple[tuple[Path, str], ...] = ()
    variant_id: str = "float"
    weight_family: str = "base-fp32"
    quantization: str = "none"
    validated_backends: tuple[str, ...] = ()
    backend_variants: tuple[Model, ...] = ()
    execution: str = "portable"
    io_precision: str = "fp32"
    feature_precision: str = "fp32"

    @classmethod
    def from_file(cls, path: str | Path) -> "Model":
        """Read an explicit JSON descriptor without inferring workspace locations.

        Args:
            path: Versioned descriptor with a full graph and/or prepared split graphs.
        """
        descriptor = Path(path).expanduser().resolve()
        try:
            data = json.loads(descriptor.read_text(encoding="utf-8"))
            if data["schema_version"] not in (1, 2) or data["alignment"] not in (64, 128):
                raise ValueError("Unsupported descriptor schema or alignment")
            if data["schema_version"] == 2:
                for field in ("variant_id", "weight_family", "quantization", "validated_backends"):
                    if field not in data:
                        raise ValueError(f"Missing prepared variant identity: {field}")
                if not isinstance(data["validated_backends"], list) or not data["validated_backends"]:
                    raise ValueError("Prepared graph families require validated backends")
            def prepared(entry: dict) -> Model:
                """Args:
                    entry: One prepared precision variant, with package-local assets.
                """
                def local(asset: dict) -> Path:
                    """Args:
                        asset: Relative graph path and declared digest.
                    """
                    path = (descriptor.parent / asset["file"]).resolve()
                    if not path.is_relative_to(descriptor.parent):
                        raise ValueError("Graphs must stay inside their descriptor directory")
                    return path
                extra = {"precision": entry.get("precision", "fp32"),
                         "external_data": tuple((local(asset), asset["sha256"]) for asset in data.get("external_data", [])),
                         "variant_id": data.get("variant_id", "float"),
                         "weight_family": data.get("weight_family", "base-fp32"),
                         "quantization": data.get("quantization", "none"),
                         "validated_backends": tuple(data.get("validated_backends", []))}
                extra.update(execution=entry.get("execution", "portable"),
                             io_precision=entry.get("io_precision", "fp32"),
                             feature_precision=entry.get("feature_precision", "fp32"))
                if "cache_graphs" in entry:
                    if set(entry["cache_graphs"]) != {"encoder", "synthesis"}:
                        raise ValueError("Both prepared cache graphs are required")
                    for role, asset in entry["cache_graphs"].items():
                        extra[role], extra[role + "_sha256"] = local(asset), asset["sha256"]
                    extra.update(encoding_channels=data["encoding_channels"], contract=entry["contract"])
                full = entry.get("graph")
                return cls(data["id"], local(full) if full else None, full["sha256"] if full else None,
                           data["alignment"], data["purpose"], **extra)
            model = prepared(data)
            if "variants" in data:
                if set(data["variants"]) != {"fp16"} or data["variants"]["fp16"].get("precision") != "fp16":
                    raise ValueError("Unsupported prepared precision variants")
                from dataclasses import replace
                model = replace(model, fp16=prepared(data["variants"]["fp16"]))
            if "backend_variants" in data:
                from dataclasses import replace
                if not isinstance(data["backend_variants"], dict) or not data["backend_variants"]:
                    raise ValueError("Prepared backend variants must be a nonempty mapping")
                variants = tuple(prepared(entry) for entry in data["backend_variants"].values())
                model = replace(model, backend_variants=variants,
                                fp16=replace(model.fp16, backend_variants=variants) if model.fp16 else None)
            model.verify()
            return model
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ModelNotInstalledError(f"Cannot read model descriptor {descriptor}: {error}") from error

    def verify(self) -> None:
        """Require verified full or split assets; never download or rewrite a fallback."""
        if (self.alignment not in (64, 128) or self.purpose not in ("reference", "production") or
                self.precision not in ("fp16", "fp32") or
                not isinstance(self.variant_id, str) or not self.variant_id or
                not isinstance(self.weight_family, str) or not self.weight_family or
                self.quantization not in ("none", "ptq", "qat") or
                self.execution not in ("portable", "trt-fused-v1") or
                self.io_precision not in ("fp16", "fp32") or self.feature_precision not in ("fp16", "fp32") or
                not isinstance(self.validated_backends, tuple) or
                any(not isinstance(name, str) or not name for name in self.validated_backends)):
            raise RifeError("Invalid model identity, graph path, alignment, or purpose")
        if self.graph is not None or self.sha256 is not None:
            if (not isinstance(self.graph, Path) or not self.graph.is_absolute() or not self.graph.is_file() or
                    not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)):
                raise RifeError("Invalid full model graph path or SHA-256")
            if file_digest(self.graph) != self.sha256:
                raise RifeError(f"Model SHA-256 mismatch: {self.graph}")
        elif self.encoder is None or self.synthesis is None:
            raise RifeError("A model requires a full graph or both prepared split graphs")
        if not isinstance(self.external_data, tuple):
            raise RifeError("External model data must be an immutable asset tuple")
        seen = set()
        for path, digest in self.external_data:
            if (not isinstance(path, Path) or not path.is_absolute() or not path.is_file() or path in seen or
                    not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or file_digest(path) != digest):
                raise RifeError(f"Invalid external model data or SHA-256 mismatch: {path}")
            seen.add(path)
        if any(v is not None for v in (self.encoder, self.synthesis, self.encoder_sha256, self.synthesis_sha256)):
            expected = "m3-trt-fused-v1" if self.execution == "trt-fused-v1" else "m4-mixed-v1" if self.precision == "fp16" else "m3-dynamic-v1"
            if self.encoding_channels not in (4, 16) or self.contract != expected:
                raise RifeError("Unsupported split-model channels or graph contract")
            for role in ("encoder", "synthesis"):
                path, digest = self.asset(role)
                if (not isinstance(path, Path) or not path.is_absolute() or not path.is_file() or
                        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or file_digest(path) != digest):
                    raise RifeError(f"Invalid prepared {role} graph or SHA-256 mismatch: {path}")
        if self.fp16 is not None:
            if (not isinstance(self.fp16, Model) or self.precision != "fp32" or self.fp16.precision != "fp16" or
                    self.fp16.fp16 is not None or (self.fp16.model_id, self.fp16.alignment, self.fp16.encoding_channels) !=
                    (self.model_id, self.alignment, self.encoding_channels) or self.fp16.external_data != self.external_data or
                    (self.fp16.variant_id, self.fp16.weight_family, self.fp16.quantization, self.fp16.validated_backends) !=
                    (self.variant_id, self.weight_family, self.quantization, self.validated_backends)):
                raise RifeError("Prepared precision variants must share one model profile")
            self.fp16.verify()
        identities = set()
        for variant in self.backend_variants:
            if (not isinstance(variant, Model) or variant.execution != "trt-fused-v1" or variant.backend_variants or
                    (variant.model_id, variant.alignment, variant.encoding_channels, variant.variant_id, variant.weight_family, variant.quantization) !=
                    (self.model_id, self.alignment, self.encoding_channels, self.variant_id, self.weight_family, self.quantization) or
                    variant.external_data != self.external_data or variant.feature_precision != variant.precision or
                    variant.validated_backends != self.validated_backends):
                raise RifeError("Invalid prepared backend layout family")
            key = (variant.execution, variant.precision, variant.io_precision)
            if key in identities:
                raise RifeError("Duplicate prepared backend layout")
            identities.add(key)
            variant.verify()

    def for_backend(self, precision: str, io_precision: str) -> Model:
        """Args:
            precision: Neural arithmetic policy already prepared in the package.
            io_precision: RGB storage layout matching the selected source format.
        """
        matches = [m for m in self.backend_variants if m.precision == precision and m.io_precision == io_precision]
        if len(matches) != 1:
            raise RifeError(f"Model {self.model_id!r} has no unique prepared TensorRT {precision}/{io_precision} layout")
        return matches[0]

    @property
    def precisions(self) -> tuple[str, ...]:
        """Return neural precision assets actually present in this model bundle."""
        return ("fp16", "fp32") if self.fp16 is not None else (self.precision,)

    def for_precision(self, precision: str) -> Model:
        """Args:
            precision: Explicit neural arithmetic policy; no runtime graph conversion.
        """
        if precision == self.precision:
            return self
        if precision == "fp16" and self.fp16 is not None:
            return self.fp16
        raise RifeError(f"Model {self.model_id!r} has no prepared {precision} asset; available={self.precisions}. "
                        "Use a precision bundle or explicitly select precision='fp32' for an FP32-only reference.")

    def asset(self, role: str) -> tuple[Path | None, str | None]:
        """Args:
            role: Full, encoder or synthesis graph already supplied by the model.
        """
        if role == "full":
            return self.graph, self.sha256
        if role not in ("encoder", "synthesis"):
            raise RifeError(f"Unknown graph role: {role}")
        return getattr(self, role), getattr(self, role + "_sha256")


def resolve_model(value: str | Model) -> Model:
    """Find an installed model provider or accept an explicit verified descriptor.

    Args:
        value: Model ID or an explicit Model instance.
    """
    if isinstance(value, Model):
        value.verify()
        return value
    matches = list(entry_points(group="rife_superboost.models", name=value))
    if len(matches) != 1:
        packages = {"4.25": "model-425", "4.25-lite": "model-425-lite", "4.26": "model-426", "4.26-large": "model-426-large",
                    "4.25-scale8": "model-425-scale8", "4.26-scale8": "model-426-scale8"}
        hint = f" Install RIFE-SuperBoost[{packages[value]}]." if value in packages else " Install its model package."
        raise ModelNotInstalledError(f"Model {value!r} is not uniquely installed.{hint} Or pass Model.from_file(...).")
    model = matches[0].load()()
    if not isinstance(model, Model):
        raise RifeError(f"Model provider {value!r} returned an incompatible identity")
    model.verify()
    return model
