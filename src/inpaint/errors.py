"""Engine-level exceptions shared by inpainters."""
from __future__ import annotations


class EngineUnavailable(RuntimeError):
    """Raised when an inpainting engine cannot be loaded on this machine.

    The pipeline factory catches this and degrades to a lighter engine so a job
    still completes instead of hard-failing.
    """
