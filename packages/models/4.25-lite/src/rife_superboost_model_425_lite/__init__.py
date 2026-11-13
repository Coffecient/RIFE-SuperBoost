# SPDX-License-Identifier: MIT
"""Installed model provider; reading data does not initialize GPU libraries."""

from pathlib import Path


def get_model():
    """Return the verified prepared bundle with its original model attribution."""
    from rife_superboost import Model

    return Model.from_file(Path(__file__).resolve().parent / "assets/model.json")
