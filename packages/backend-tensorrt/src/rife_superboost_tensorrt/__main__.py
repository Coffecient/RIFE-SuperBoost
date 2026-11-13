# SPDX-License-Identifier: MIT
"""Explicit GPU/runtime probe: python -m rife_superboost_tensorrt."""

import argparse
import json
import sys

from ._loader import load_native, resolve_runtime


def main() -> int:
    """Probe library origins and the selected device without building an engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", help="Explicit NVIDIA DLL directory; defaults to the pinned installed wheel.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal to inspect.")
    args = parser.parse_args()
    try:
        native = load_native(resolve_runtime(args.runtime_dir))
        report = {"status": "ready", "runtime_dir": str(native.runtime_dir),
                  "native": native.identity, "device": native.device(args.device)}
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
