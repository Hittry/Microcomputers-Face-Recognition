import cv2
import numpy as np
import torch.nn
from numpy.typing import NDArray
from torchvision.transforms import ToTensor


class FaceRecognizer:
    """Класс для работы с моделью распознавания лиц."""

    def __init__(self, model_path: str, image_size: int = 160, device_type: str = "cpu") -> None:
        self.image_size = image_size
        self.device_type = device_type

        self.model = self._prepare_model(model_path)

    def _prepare_model(self, model_path: str) -> torch.nn.Module:
        """
        Метод для подготовки модели распознавания.

        Args:
            model_path: Путь до модели распознавания.

        Returns:
            Модель распознавания.
        """

        model = torch.jit.load(model_path, map_location=torch.device(self.device_type))
        model = torch.jit.optimize_for_inference(model.eval())
        return model

    def _prepare_transform(self, face_box: NDArray[np.float32]) -> torch.Tensor:
        """
        Метод для преобразования изображения для модели.

        Args:
            face_box: Изображение в RGB формате.

        Returns:
            Tensor для модели.
        """

        face_box = cv2.resize(
            face_box,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        face_box = np.ascontiguousarray(face_box).astype(np.float32)
        face_box -= 127.5  # с [0; 255] на [-127.5; 127.5]
        face_box /= 128.0  # с [-127.5; 127.5] на [-1.0; 1.0]
        face = ToTensor()(face_box)[None]
        return face

    def predict(
        self,
        face_box: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Метод для получения дескрипторов лиц.

        Args:
            face_box: Изображение лица для анализа. Формат RGB.

        Returns:
            Дескрипторы лиц.
        """

        with torch.inference_mode():
            transformed_image = self._prepare_transform(face_box)
            prediction = self.model(transformed_image.to(self.device_type))
        return prediction.detach().cpu().numpy()[0]
