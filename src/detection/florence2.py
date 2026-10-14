"""Florence-2 based automatic watermark detection (Microsoft vision model).

Uses the open-vocabulary ``<OD>`` (open detection) task with a configurable text
prompt (default ``"watermark"``) to locate watermark regions without manual
selection, mirroring the approach of D-Ogi/WatermarkRemover-AI.

Heavy / optional: requires ``transformers`` and downloads the Florence-2
checkpoint from the HuggingFace hub on first use.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

from src.core.types import BBox, DetectionMethod, DetectionResult, Mask2D
from src.detection.base import Detector, boxes_to_mask
from src.utils.device import DeviceInfo
from src.utils.logger import get_logger

_logger = get_logger("florence2")


class Florence2Detector(Detector):
    method = DetectionMethod.FLORENCE2
    requires_model = True

    def __init__(self, device: DeviceInfo, model_id: str = "microsoft/Florence-2-base",
                 prompt: str = "watermark", max_bbox_percent: float = 25.0,
                 min_confidence: float = 0.1) -> None:
        self.device = device
        self.model_id = model_id
        self.prompt = prompt
        self.max_bbox_percent = max_bbox_percent
        self.min_confidence = min_confidence
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Florence2ForConditionalGeneration as Model  # type: ignore
        except Exception:  # noqa: BLE001 - older transformers
            from transformers import AutoModelForCausalLM as Model  # type: ignore

        _logger.info("Loading Florence-2 (%s) on %s", self.model_id, self.device.torch_device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = Model.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.is_cuda else torch.float32,
            attn_implementation="sdpa",
        ).to(self.device.torch_device).eval()

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self._model is None:
            self.load()
        import torch

        h, w = image.shape[:2]
        rgb = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        task = "<OD>"
        inputs = self._processor(text=task, images=rgb, return_tensors="pt")
        inputs = inputs.to(self.device.torch_device, torch.float16 if self.device.is_cuda else torch.float32)

        with torch.no_grad():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
            )

        decoded = generated[..., 0] if generated.ndim == 3 else generated
        parsed = self._processor.batch_post_process_generation(decoded, task, inputs, [task])
        result = parsed[0].get(task, {}) if parsed else {}
        labels = result.get("labels", [])
        bboxes = result.get("bboxes", [])
        confs = result.get("scores", [1.0] * len(bboxes))

        boxes: List[BBox] = []
        frame_area = float(h * w)
        for box, score in zip(bboxes, confs):
            if score is not None and float(score) < self.min_confidence:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            # Florence-2 returns coordinates normalized to [0, 1]
            px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
            bbox = BBox(px0, py0, px1, py1)
            if frame_area > 0 and (bbox.area / frame_area) * 100.0 > self.max_bbox_percent:
                continue
            boxes.append(bbox)

        mask = boxes_to_mask((h, w), boxes, dilate=2) if boxes else np.zeros((h, w), np.uint8)
        _logger.info("Florence-2 (%s) found %d candidate region(s)", self.prompt, len(boxes))
        return DetectionResult(masks=[Mask2D(data=mask, boxes=boxes)] if boxes else [],
                               method=self.method)
