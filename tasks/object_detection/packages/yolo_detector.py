"""ONNX YOLO detector for the duckiebot.

Loads a YOLOv5- or YOLOv8-exported ONNX model and returns detections in the
*original* camera-image pixel space, so the filters in integration_activity
(which are written against the camera's 640x480 frame) can use bbox pixel
coordinates directly.

Output format detection is automatic — both v5 and v8 ONNX export shapes are
handled.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

Detection = Tuple[Tuple[int, int, int, int], float, int]


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        input_size: Optional[int] = None,
        conf_thres: float = 0.25,
        nms_thres: float = 0.45,
        num_classes: int = 3,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            ) from e

        self.conf_thres = float(conf_thres)
        self.nms_thres = float(nms_thres)
        self.num_classes = int(num_classes)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Read the model's expected square input size from its shape.
        ishape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
        inferred = next((d for d in ishape[2:]
                         if isinstance(d, int) and d > 0), None)
        self.input_size = int(input_size if input_size is not None
                              else (inferred or 640))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        """Run inference on a BGR image. Returns [(bbox, score, class_id), ...]
        with bboxes in the original image pixel coords (xmin, ymin, xmax, ymax)."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        h0, w0 = image_bgr.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(image_bgr, self.input_size)

        rgb = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = rgb.transpose(2, 0, 1)[None]  # NCHW

        outputs = self.session.run(None, {self.input_name: tensor})
        pred = outputs[0]  # (1, N, 5+nc) for v5 or (1, 4+nc, N) for v8

        boxes_xywh, scores, class_ids = self._decode(pred)
        if boxes_xywh.size == 0:
            return []

        # xywh (center) -> xyxy in letterboxed image coords
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Undo letterbox -> original-image coords
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale

        x1 = np.clip(x1, 0, w0 - 1)
        x2 = np.clip(x2, 0, w0 - 1)
        y1 = np.clip(y1, 0, h0 - 1)
        y2 = np.clip(y2, 0, h0 - 1)

        # Class-aware NMS via OpenCV
        rects = [[int(x1[i]), int(y1[i]),
                  int(x2[i] - x1[i]), int(y2[i] - y1[i])]
                 for i in range(len(scores))]
        idxs = cv2.dnn.NMSBoxes(rects, scores.tolist(),
                                self.conf_thres, self.nms_thres)

        detections: List[Detection] = []
        if len(idxs) > 0:
            idxs = np.array(idxs).flatten()
            for i in idxs:
                detections.append((
                    (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                    float(scores[i]),
                    int(class_ids[i]),
                ))
        return detections

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode(self, pred: np.ndarray):
        """Return (boxes_xywh, scores, class_ids), filtered by conf_thres."""
        arr = pred[0] if pred.ndim == 3 else pred

        # v8 shape: (4+nc, N) — channels < detections
        # v5 shape: (N, 5+nc) — detections > channels
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T  # -> (N, 4+nc)
            boxes = arr[:, :4]
            cls_scores = arr[:, 4:]
            cls_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
        else:
            # v5: (N, 5+nc) = [cx, cy, w, h, obj, cls...]
            boxes = arr[:, :4]
            obj = arr[:, 4]
            cls_scores = arr[:, 5:]
            cls_ids = np.argmax(cls_scores, axis=1)
            scores = obj * cls_scores[np.arange(len(cls_ids)), cls_ids]

        mask = scores >= self.conf_thres
        return boxes[mask], scores[mask], cls_ids[mask]

    @staticmethod
    def _letterbox(image: np.ndarray, size: int):
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y
