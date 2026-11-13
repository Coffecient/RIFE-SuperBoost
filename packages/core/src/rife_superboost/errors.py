# SPDX-License-Identifier: MIT
"""Actionable errors shared by the core and optional backends."""


class RifeError(RuntimeError):
    """Base error for model discovery, configuration, and execution."""


class ModelNotInstalledError(RifeError):
    """The selected model package or explicit graph is unavailable."""


class BackendNotInstalledError(RifeError):
    """The optional backend distribution is not installed."""


class DependencyError(RifeError):
    """A native dependency is absent or cannot be loaded."""


class VersionConflictError(RifeError):
    """The process already contains incompatible native libraries."""


class EngineBuildError(RifeError):
    """An isolated engine build failed; its log is retained."""


class InferenceError(RifeError):
    """A native execution request failed."""
