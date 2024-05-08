from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from face_detection.helper_functions import HelpPostProcessFunctions, ImageScale


class FaceDetector:
    """Класс для детектирования лиц."""

    def __init__(
        self,
        model_path: str,
        iou_threshold: float = 0.3,
        device_type: str = "cpu",
    ) -> None:
        self.postprocess_functions = HelpPostProcessFunctions()

        self.iou_threshold = iou_threshold

        self.device_type = device_type
        self.model = self._prepare_model(model_path)

    def _prepare_transform(
        self,
        frame: NDArray[np.float32],
        image_w: int,
        image_h: int,
    ) -> torch.Tensor:
        """
        Подготовка к преобразованию изображения.

        Args:
            frame: Кадр для детектирования лиц. В BGR формате.
            image_w: Ширина изображения.
            image_h: Высота изображения.

        Returns:
            Tensor для передачи в модель детектирования.
        """

        blob = cv2.dnn.blobFromImage(
            frame,
            size=(image_w, image_h),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        blob = torch.from_numpy(blob)
        return blob

    def _prepare_model(self, model_path: str) -> torch.nn.Module:
        """
        Подготовка модели.

        Args:
            model_path: Путь до модели.

        Returns:
            Модель детектирования лиц.
        """

        model = torch.jit.load(model_path, map_location=torch.device(self.device_type))
        model = torch.jit.optimize_for_inference(model.eval())
        return model

    def predict(
        self,
        frame: NDArray[np.float32],
        conf_threshold: float = 0.5,
    ) -> tuple[Any, Any, Any]:
        """
        Метод для детектирования лиц на изображении.

        Args:
            frame: Кадр для детектирования. В формате BGR.
            conf_threshold: Уверенность модели в детектировании для фильтрации.

        Returns:
            Список боксов и уверенностей модели.
        """

        scaled_image = self.__prepare_frame_scale(frame)
        blob = self._prepare_transform(
            frame=frame,
            image_w=scaled_image.image_w,
            image_h=scaled_image.image_h,
        )
        with torch.inference_mode():
            heatmap, scale, offset, lms = self.model(blob)

        heatmap = heatmap.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        offset = offset.detach().cpu().numpy()
        lms = lms.detach().cpu().numpy()

        boxes_pred, landmarks = self.postprocess_functions.postprocess(
            heatmap=heatmap,
            scale=scale,
            offset=offset,
            lms=lms,
            scale_image_info=scaled_image,
            conf_threshold=conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        lms_res_list = []
        if landmarks.shape[0]:
            for lms_res in landmarks:
                split_res = []
                for i in range(5):
                    split_res.append(
                        [int(lms_res[i * 2]), int(lms_res[i * 2 + 1])]
                    )
                lms_res_list.append(split_res)
        return (
            boxes_pred[:, :4].astype(int).tolist(),  # boxes
            boxes_pred[:, 4].tolist(),  # scores
            lms_res_list,  # landmarks
        )

    @staticmethod
    def __prepare_frame_scale(frame: NDArray[np.float32]) -> ImageScale:
        """
        Метод для изменения масштабов изображения.

        Args:
            frame: Кадр для детектирования. В формате BGR.

        Returns:
            Новый параметры изображения после преобразования.
        """

        orig_frame_h, orig_frame_w = frame.shape[:2]
        new_frame_h = int(np.ceil(orig_frame_h / 32) * 32)
        new_frame_w = int(np.ceil(orig_frame_w / 32) * 32)
        scale_h = new_frame_h / orig_frame_h
        scale_w = new_frame_w / orig_frame_w
        result = ImageScale(
            image_h=new_frame_h,
            image_w=new_frame_w,
            image_scale_h=scale_h,
            image_scale_w=scale_w,
        )
        return result
