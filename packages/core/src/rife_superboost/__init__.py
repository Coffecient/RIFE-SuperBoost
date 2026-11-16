# SPDX-License-Identifier: MIT
"""VapourSynth interpolation core; optional GPU dependencies are loaded lazily."""

from .api import capabilities, interpolate
from .color import convert_color
from .models import Model
from .errors import RifeError

__version__ = "0.1.0.dev9"
__all__ = ["Model", "RifeError", "capabilities", "convert_color", "interpolate"]
