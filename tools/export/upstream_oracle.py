# SPDX-License-Identifier: MIT
"""Execute only hash-verified upstream inference definitions for comparison.

The official files remain byte-for-byte unchanged in models/assets/. Imports of
the upstream package layout are removed from the AST and the official warp is
injected into a private namespace. The original inference method supplies scales.
"""

from __future__ import annotations

import ast
from types import MethodType, SimpleNamespace
from typing import Any

import torch

from assets import MANIFEST, load_checkpoint, verify_file


def build_oracle(model: dict[str, Any], device: str = "cpu") -> SimpleNamespace:
    """Load the original network and inference method without training imports.

    Args:
        model: Pinned model entry; every executed source is hash-verified first.
        device: Explicit torch device, also used by the original grid cache.
    """
    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    warp_record = next(item for item in manifest["upstream_snapshots"] if item["path"].endswith("warplayer.upstream.py"))
    warp_path = verify_file(warp_record)
    warp_namespace = {"__name__": "rife_superboost_official_warp"}
    exec(compile(warp_path.read_text(encoding="utf-8"), str(warp_path), "exec"), warp_namespace)
    warp_namespace["device"] = torch.device(device)

    network_record = next(item for item in model["source_files"] if item["path"].endswith("/IFNet_HDv3.py"))
    network_path = verify_file(network_record)
    tree = ast.parse(network_path.read_text(encoding="utf-8"), filename=str(network_path))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "model.warplayer"]
    if len(imports) != 1 or ast.unparse(imports[0]) != "from model.warplayer import warp":
        raise ValueError("Unexpected official warp import; review the pinned source.")
    tree.body.remove(imports[0])
    namespace = {"__name__": "rife_superboost_official_network", "warp": warp_namespace["warp"]}
    exec(compile(tree, str(network_path), "exec"), namespace)
    network = namespace["IFNet"]().eval().requires_grad_(False)
    state, _ = load_checkpoint(model)
    network.load_state_dict(state, strict=True)
    network.to(device)

    entry_record = next(item for item in model["source_files"] if item["path"].endswith("/RIFE_HDv3.py"))
    entry_path = verify_file(entry_record)
    entry_tree = ast.parse(entry_path.read_text(encoding="utf-8"), filename=str(entry_path))
    model_class = next(node for node in entry_tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    method = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "inference")
    entry_namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(entry_path), "exec"), entry_namespace)
    oracle = SimpleNamespace(flownet=network, warp=warp_namespace["warp"], grid_cache=warp_namespace["backwarp_tenGrid"])
    oracle.inference = MethodType(entry_namespace["inference"], oracle)
    return oracle
