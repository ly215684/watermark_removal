"""LaMa inpainting via the official ``big-lama.pt`` TorchScript checkpoint.

The model takes a normalized RGB float tensor and a binary mask (1 = hole) and
returns a reconstructed image. Images are padded to a multiple of 8 and the
padding removed afterwards.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.core.types import Progress
from src.inpaint.base import Inpainter
from src.models import downloader
from src.utils.device import DeviceInfo
from src.utils.logger import get_logger

_logger = get_logger("lama")

_PAD = 8


class LamaInpainter(Inpainter):
    name = "lama"
    supports_temporal = False
    weight_key = "lama"

    def __init__(self, device: DeviceInfo, super_fast: bool = False) -> None:
        super().__init__(device)
        self.super_fast = super_fast
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        path = downloader.ensure("lama")
        _logger.info("Loading LaMa from %s on %s", path, self.device.torch_device)
        self._model = torch.jit.load(str(path), map_location=self.device.torch_device)
        self._model.eval()

    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Nothing to do.
        if mask is None or not np.any(mask):
            return image
        import torch

        if self._model is None:
            self.load()

        h, w = image.shape[:2]
        # BGR -> RGB, normalized to [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # hole mask: nonzero -> 1
        m = (mask > 0).astype(np.float32)
        if self.super_fast:
            rgb, m, _ = self._downscale(rgb, m)
        # content size to crop back to after padding (post any downscale)
        out_h, out_w = int(rgb.shape[0]), int(rgb.shape[1])

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        m_t = torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0)

        rgb_p, m_p, (ph, pw) = self._pad_to_multiple(rgb_t, m_t)
        batch_img = rgb_p.unsqueeze(0).to(self.device.torch_device)
        batch_msk = m_p.unsqueeze(0).to(self.device.torch_device)

        with torch.no_grad():
            out = self._model(batch_img, batch_msk)

        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        # remove padding, then restore original resolution if downscaled
        out = out[:min(ph, out_h), :min(pw, out_w)]
        if self.super_fast:
            out = cv2.resize(out, (w, h))

        result = (out * 255.0).round().astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # Keep original outside the hole to avoid edge color shift.
        keep = (mask == 0)
        blended = image.copy()
        blended[keep] = result[keep]
        return blended

    # helpers ---------------------------------------------------------------
    @staticmethod
    def _pad_to_multiple(img, msk):
        _, h, w = img.shape
        ph = ((h - 1) // _PAD + 1) * _PAD
        pw = ((w - 1) // _PAD + 1) * _PAD
        pad_img = ((0, pw - w), (0, ph - h))
        import torch.nn.functional as F

        img = F.pad(img, pad_img)
        msk = F.pad(msk, pad_img)
        return img, msk, (ph, pw)

    @staticmethod
    def _downscale(rgb, m, max_side=800):
        h, w = rgb.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale >= 1.0:
            return rgb, m, (h, w)
        nh, nw = int(h * scale), int(w * scale)
        rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        m_s = cv2.resize((m * 255).astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) / 255.0
        return rgb_s, m_s.astype(np.float32), (h, w)

    def progress_stub(self) -> Progress:  # pragma: no cover - helper for typing
        return Progress(value=0)
