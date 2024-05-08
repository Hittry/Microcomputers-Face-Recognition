from datetime import datetime
from pathlib import Path

import numpy as np
from imagecodecs import jpegxl_encode
from numpy.typing import NDArray

from const import BASE_DIR


class ImageProc:

    def __init__(self, incidents_save_dir: str) -> None:
        self.incidents_save_dir = incidents_save_dir

    def write_encode_photo(
        self,
        photo_id: str,
        frame_for_save: NDArray[NDArray[np.float32]],
        compress_quality: int,
        compress_effort: int,
    ) -> None:
        """
        Метод для сохранения лиц с инцидентами в JXL.

        Args:
            photo_id: ID фото инцидента
            frame_for_save: Изображения в RGB формате.
            compress_quality: Качество изображения после сжатия.
            compress_effort: Уровень сжатия, чем больше тем больше сжатие, но дольше.
        """

        curr_dt = datetime.now()
        timestamp = str(round(curr_dt.timestamp()))
        jpeg_xl_enc = jpegxl_encode(
            frame_for_save,
            level=compress_quality,
            lossless=False,
            effort=compress_effort,
        )
        Path(BASE_DIR / self.incidents_save_dir / timestamp).mkdir(exist_ok=True, parents=True)
        frame_path_for_save = BASE_DIR / self.incidents_save_dir / timestamp / f"{photo_id}.jxl"
        with open(str(frame_path_for_save), "wb") as binary_file:
            binary_file.write(jpeg_xl_enc)
