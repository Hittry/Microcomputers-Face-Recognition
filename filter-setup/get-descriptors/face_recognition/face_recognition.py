import numpy as np
import torch.nn
from PIL import Image
from numpy.typing import NDArray
from torchvision import transforms


class FaceRecognition:

    def __init__(self, model_path: str, device_type: str = "cpu") -> None:
        self.device_type = device_type

        self.model = self._prepare_model(model_path)
        self.image_transform = self._prepare_transform()

    def _prepare_model(self, model_path: str) -> torch.nn.Module:
        """Метод для инициализации модели."""

        model = torch.jit.load(model_path, map_location=torch.device(self.device_type))
        model = torch.jit.optimize_for_inference(model.eval())
        return model

    def _prepare_transform(self) -> transforms.Compose:
        """Метод для подготовки изображения к входу модели."""

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def predict(
        self,
        face_box_rgb: NDArray[NDArray[np.float32]],
    ) -> NDArray[NDArray[np.float32]]:
        """
        Метод для предсказания модели.

        Args:
            face_box_rgb: Изображение лица в RGB формате.

        Returns:
            Дескриптор лица.
        """

        face_rgb_pil = Image.fromarray(face_box_rgb).resize((112, 112))
        transformed_input = self.image_transform(face_rgb_pil).unsqueeze(0)
        with torch.inference_mode():
            result = self.model(transformed_input)
        result = result.detach().numpy()
        return result
