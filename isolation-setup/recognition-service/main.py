from argparse import ArgumentParser
from pathlib import Path
from time import sleep
from uuid import uuid4

import cv2
from loguru import logger

from config import get_mapping_config, get_service_config
from const import BASE_DIR
from face_align import FaceAlign
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from faiss_manager import FaissManager
from image_processing import ImageProc
from logs import prepare_logging


def main(log_dir: str, inc_photos_dir: str, cap_id: int) -> None:
    prepare_logging(log_dir)
    Path(BASE_DIR / inc_photos_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Чтение конфигураций")
    service_config = get_service_config()
    mapping_config = get_mapping_config()
    logger.info("Конфигурации успешно получены")

    logger.info("Подготовка моделей")
    faiss_mgr = FaissManager(str(BASE_DIR / "dist" / "faiss_rec.index"))
    image_proc = ImageProc(inc_photos_dir)
    face_align_method = FaceAlign()
    face_detection_model = FaceDetector(str(BASE_DIR / "dist" / "centerface_scripted.pt"))
    face_recognition_model = FaceRecognizer(str(BASE_DIR / "dist" / "face_recognition.pt"))
    logger.info("Инициализация моделей завершена успешно")

    cap = cv2.VideoCapture(cap_id)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Не удалось подключиться к камере с id: `{args.cap}`")
            raise Exception("Error when webcam connection")

        logger.info("Детектирование лиц")
        boxes, _, faces_landmarks = face_detection_model.predict(frame)
        if len(boxes) != 1:
            logger.info("Нет лиц в кадре")
            continue
        logger.info(f"Лица успешно получены. Число лиц: `{len(boxes)}`")

        is_save_inc = False
        for face_lms in faces_landmarks:
            face_image = face_align_method.warp_and_crop_face(frame, face_lms)
            face_box_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            emb = face_recognition_model.predict(face_box_rgb)
            sim_index = faiss_mgr.search_sim_vector_index(
                emb, threshold=service_config["known_face_search_threshold"]
            )
            if sim_index is None:
                logger.error("ALERT!!! Неизвестный пользователь")
                if not is_save_inc:
                    inc_uuid = uuid4()
                    image_proc.write_encode_photo(
                        photo_id=str(inc_uuid),
                        frame_for_save=frame,
                        compress_quality=service_config["compress_level_quality"],
                        compress_effort=service_config["compress_effort"],
                    )
                    logger.error(f"Фото сохранено с uuid: `{str(inc_uuid)}`")
                    is_save_inc = True
                continue

            known_person_name = ""
            for person_name, indexes in mapping_config.items():
                if sim_index in indexes:
                    known_person_name = person_name
                    break
            logger.info(f"Обнаружен пользователь: `{known_person_name}`")

        sleep(service_config["sleep_time"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log", type=str, help="Path to dir with logs", dest="log", default="./logs")
    parser.add_argument("--cap", type=int, help="Webcam id", dest="cap", default=0)
    parser.add_argument("--photos", type=str, help="Path to dir with incidents", dest="photos", default="photos")
    args = parser.parse_args()

    main(args.log, args.photos, args.cap)
