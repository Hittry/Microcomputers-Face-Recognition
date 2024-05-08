from argparse import ArgumentParser

import cv2
import httpx
from loguru import logger

from config import get_service_config, get_mapping_config
from const import BASE_DIR
from face_align import FaceAlign
from face_detection import FaceDetector
from face_recognition import FaceRecognition
from faiss_manager import FaissManager
from image_processing import ImageProc
from logs import prepare_logging


def main(log_dir: str, cap_id: int, host: str, port: int, endpoint: str) -> None:
    prepare_logging(log_dir)

    service_config = get_service_config()
    mapping_config = get_mapping_config()

    logger.info("Инициализация моделей")
    faiss_mgr = FaissManager(str(BASE_DIR / "dist" / "edge.index"))
    face_align_method = FaceAlign()
    face_detection_model = FaceDetector(str(BASE_DIR / "dist" / "centerface_scripted.pt"))
    face_recognition_model = FaceRecognition(str(BASE_DIR / "dist" / "edge_face_s.scripted"))
    image_proc = ImageProc()
    logger.info("Модели успешно инициализированы")

    with httpx.Client() as client:
        cap = cv2.VideoCapture(args.cap)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Не удалось подключиться к камере с id: `{cap_id}`")
                continue

            boxes, _, faces_landmarks = face_detection_model.predict(frame)
            if len(boxes) != 1:
                logger.info("Нет лиц в кадре")
                continue
            logger.info(f"Лица успешно получены. Число лиц: `{len(boxes)}`")

            is_send_incorrect = False
            for face_lms in faces_landmarks:
                face_image = face_align_method.warp_and_crop_face(frame, face_lms)
                face_box_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                emb = face_recognition_model.predict(face_box_rgb)
                sim_index = faiss_mgr.search_sim_vector_index(
                    emb, threshold=service_config["known_face_search_threshold"]
                )
                if sim_index is None:
                    logger.error("ALERT!!! Неизвестный пользователь")
                    if not is_send_incorrect:
                        is_correct_send = image_proc.send_encode_frame(
                            client=client,
                            address_for_send=f"http://{host}:{port}{endpoint}",
                            frame_for_send=frame,
                            compress_quality=service_config["compress_level_quality"],
                            compress_effort=service_config["compress_effort"],
                        )
                        if not is_correct_send:
                            logger.error("Ошибка отправки инцидента на сервер")
                            continue

                        logger.error("Фото успешно отправлено на сервер для повторной проверки")
                        is_send_incorrect = True
                    continue

                known_person_name = ""
                for person_name, indexes in mapping_config.items():
                    if sim_index in indexes:
                        known_person_name = person_name
                        break
                logger.info(f"Обнаружен пользователь: `{known_person_name}`")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log", type=str, help="Path to dir with logs", dest="log", default="logs")
    parser.add_argument("--cap", type=int, help="Webcam id", dest="cap", default=0)
    parser.add_argument("--port", type=int, help="Port for send photos", dest="port", default=8000)
    parser.add_argument("--host", type=str, help="Host for send photos", dest="host", default="127.0.0.1")
    parser.add_argument("--end", type=str, help="Endpoint for req", dest="endpoint", default="/")
    args = parser.parse_args()

    main(
        log_dir=args.log,
        cap_id=args.cap,
        host=args.host,
        port=args.port,
        endpoint=args.endpoint,
    )
