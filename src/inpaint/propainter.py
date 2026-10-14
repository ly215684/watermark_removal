"""ProPainter temporal inpainting engine.

ProPainter (Zhou et al., CVPR'23) gives the strongest temporal consistency for
fast-motion footage, at the cost of high VRAM and slower runs.

Like STTN, the upstream architecture is expected to be vendored at
``src/inpaint/_vendor/propainter_impl.py`` exposing::

    def load_model(weights_path: str, device: str) -> \
        Callable[[list[np.ndarray], list[np.ndarray]], list[np.ndarray]]

i.e. a callable taking (frames, per-frame masks) and returning repaired frames.
When the module is missing, :meth:`load` raises :class:`EngineUnavailable` and
the pipeline falls back to LaMa. See README "Temporal engines".
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.inpaint.base import Inpainter
from src.inpaint.errors import EngineUnavailable
from src.core.types import Progress
from src.models import downloader
from src.utils.logger import get_logger

_logger = get_logger("propainter")


class ProPainterInpainter(Inpainter):
    name = "propainter"
    supports_temporal = True
    weight_key = "propainter"

    def __init__(self, device, neighbor_length: int = 10, refstride: int = 10) -> None:  # noqa: ANN001
        super().__init__(device)
        self.neighbor_length = neighbor_length
        self.refstride = refstride
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from src.inpaint._vendor import propainter_impl  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(
                "ProPainter requires the vendored model module "
                "src/inpaint/_vendor/propainter_impl.py (see README)."
            ) from exc
        weights = downloader.ensure("propainter")
        self._model = propainter_impl.load_model(str(weights), self.device.torch_device)
        _logger.info("ProPainter loaded on %s", self.device.torch_device)

    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ProPainter is temporal; use inpaint_video")

    def inpaint_video(self, frames, masks, progress=None, cancel=None) -> List[np.ndarray]:
        if self._model is None:
            self.load()
        if cancel and cancel():
            return list(frames)
        results = self._model(list(frames), list(masks))
        if progress:
            progress(Progress(value=100, current=len(frames), total=len(frames),
                              message="ProPainter done"))
        return list(results)
