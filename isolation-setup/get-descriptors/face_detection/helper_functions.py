from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ImageScale:
    """Класс данных измененного изображения."""

    image_h: int
    image_w: int
    image_scale_h: float
    image_scale_w: float


class HelpPostProcessFunctions:
    """Класс вспомогательных методов для детектирования лиц."""

    def __init__(self) -> None:
        ...

    def postprocess(
        self,
        heatmap: NDArray[NDArray[np.float32]],
        scale: NDArray[NDArray[np.float32]],
        offset: NDArray[NDArray[np.float32]],
        lms: NDArray[NDArray[np.float32]],
        scale_image_info: ImageScale,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.3,
    ) -> tuple[NDArray[NDArray[np.float32]], NDArray[NDArray[np.float32]]]:
        detections, landmarks = self._decode_predict(
            heatmap=heatmap,
            scale=scale,
            offset=offset,
            landmark=lms,
            image_size=(scale_image_info.image_h, scale_image_info.image_w),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        if not len(detections):
            detections = np.empty(shape=[0, 5], dtype=np.float32)
            landmarks = np.empty(shape=[0, 10], dtype=np.float32)
            return detections, landmarks

        detections[:, 0:4:2] /= scale_image_info.image_scale_w
        detections[:, 1:4:2] /= scale_image_info.image_scale_h
        landmarks[:, 0:10:2] /= scale_image_info.image_scale_w
        landmarks[:, 1:10:2] /= scale_image_info.image_scale_h
        return detections, landmarks

    def _decode_predict(
        self,
        heatmap: NDArray[NDArray[np.float32]],
        scale: NDArray[NDArray[np.float32]],
        offset: NDArray[NDArray[np.float32]],
        landmark: NDArray[NDArray[np.float32]],
        image_size: tuple[int, int],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.3,
    ) -> NDArray[NDArray[np.float32]] | NDArray:
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > conf_threshold)

        boxes, lms = [], []
        if len(c0) <= 0:
            return np.array(boxes, dtype=np.float32), np.array(lms, dtype=np.float32)

        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2)
            y1 = max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, image_size[1]), min(y1, image_size[0])
            boxes.append([x1, y1, min(x1 + s1, image_size[1]), min(y1 + s0, image_size[0]), s])

            lm = []
            for j in range(5):
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)

        boxes = np.asarray(boxes, dtype=np.float32)
        # Реализация через torchvision работает дольше
        keep = self._nms(
            boxes=boxes[:, :4],
            scores=boxes[:, 4],
            iou_threshold=iou_threshold,
        )
        boxes = boxes[keep, :]
        lms = np.asarray(lms, dtype=np.float32)
        lms = lms[keep, :]
        return boxes, lms

    @staticmethod
    def _nms(
        boxes: NDArray[NDArray[np.float32]],
        scores: NDArray[np.float32],
        iou_threshold: float = 0.3,
    ) -> list[np.int64]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool_)
        keep = []
        for detection_index in range(num_detections):
            sorted_index = order[detection_index]
            if suppressed[sorted_index]:
                continue

            keep.append(sorted_index)
            ix1 = x1[sorted_index]
            iy1 = y1[sorted_index]
            ix2 = x2[sorted_index]
            iy2 = y2[sorted_index]
            iarea = areas[sorted_index]

            for next_det_index in range(detection_index + 1, num_detections):
                next_sorted_index = order[next_det_index]
                if suppressed[next_sorted_index]:
                    continue

                xx1 = max(ix1, x1[next_sorted_index])
                yy1 = max(iy1, y1[next_sorted_index])
                xx2 = min(ix2, x2[next_sorted_index])
                yy2 = min(iy2, y2[next_sorted_index])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[next_sorted_index] - inter)
                if ovr >= iou_threshold:
                    suppressed[next_sorted_index] = True
        return keep
