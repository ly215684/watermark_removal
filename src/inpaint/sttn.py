"""STTN (Spatial-Temporal Transformer Network) temporal inpainting engine.

STTN reconstructs masked video regions using information from neighbouring
frames, giving temporally consistent results for real-world footage.

Design note
-----------
The upstream network definition (ViT encoder blocks, attention module and
decoder) and its official checkpoint come from
``YaoFANGUK/video-subtitle-remover`` / ``microsoft/StTN``. To avoid shipping a
duplicated, unverifiable copy of a large third-party model inside this repo, the
architecture is expected to be provided as a *vendored* module:

    src/inpaint/_vendor/sttn_impl.py

That module must expose::

    def load_model(weights_path: str, device: str) -> Callable[[list[np.ndarray], np.ndarray], list[np.ndarray]]

If the vendored module is absent, :meth:`STTNInpainter.load` raises
``EngineUnavailable`` and the pipeline factory transparently falls back to
per-frame LaMa so the job still completes. See README "Temporal engines".
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.core.types import CancelCheck, Progress
from src.inpaint.base import Inpainter
from src.inpaint.errors import EngineUnavailable
from src.models import downloader
from src.utils.logger import get_logger

_logger = get_logger("sttn")


class STTNInpainter(Inpainter):
    name = "sttn"
    supports_temporal = True
    weight_key = "sttn"

    def __init__(self, device, neighbor_stride: int = 10, reference_length: int = 10,  # noqa: ANN001
                 max_load_num: int = 30) -> None:
        super().__init__(device)
        self.neighbor_stride = neighbor_stride
        self.reference_length = reference_length
        self.max_load_num = max_load_num
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from src.inpaint._vendor import sttn_impl  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(
                "STTN requires the vendored model module "
                "src/inpaint/_vendor/sttn_impl.py (see README)."
            ) from exc
        weights = downloader.ensure("sttn")
        self._model = sttn_impl.load_model(str(weights), self.device.torch_device)
        _logger.info("STTN loaded on %s", self.device.torch_device)

    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("STTN is temporal; use inpaint_video")

    def inpaint_video(self, frames, masks, progress=None, cancel=None) -> List[np.ndarray]:
        if self._model is None:
            self.load()
        # STTN works on a shared (usually constant) mask across the clip; take the
        # union so temporally-moving watermarks are all covered per window.
        results: List[np.ndarray] = list(frames)
        total = len(frames)
        step = self.max_load_num
        for start in range(0, total, step):
            if cancel and cancel():
                break
            end = min(total, start + step)
            window = frames[start:end]
            union = np.zeros_like(masks[0])
            for m in masks[start:end]:
                union = np.maximum(union, m)
            filled = self._model(window, union)
            for i, f in enumerate(filled):
                results[start + i] = f
            if progress:
                progress(Progress(value=int(100 * end / total), current=end, total=total,
                                  message=f"STTN {end}/{total}"))
        return results
