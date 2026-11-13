# SPDX-License-Identifier: MIT
"""Optional TensorRT backend; importing this module does not load GPU DLLs."""

from .backend import TensorRT

__all__ = ["TensorRT"]
__version__ = "0.1.0.dev8"
