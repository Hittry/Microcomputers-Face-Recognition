import base64

import httpx
import numpy as np
from imagecodecs import jpegxl_encode
from numpy.typing import NDArray


class ImageProc:

    def __init__(self) -> None:
        ...

    def send_encode_frame(
        self,
        client: httpx.Client,
        address_for_send: str,
        frame_for_send: NDArray[NDArray[np.float32]],
        compress_quality: int,
        compress_effort: int,
    ) -> bool:

        jpeg_xl_enc = jpegxl_encode(
            frame_for_send,
            level=compress_quality,
            lossless=False,
            effort=compress_effort,
        )
        encoded_string = base64.b64encode(jpeg_xl_enc).decode("UTF-8")
        data = {"image": encoded_string}
        resp = client.post(address_for_send, json=data)
        if resp.status_code == 202:
            return True
        return False
