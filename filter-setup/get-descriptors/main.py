import json
import os
from argparse import ArgumentParser

import cv2

from const import BASE_DIR
from face_align import FaceAlign
from face_detection import FaceDetector
from face_recognition import FaceRecognition
from faiss_manager import FaissManager


def main(photos_path: str):
    faiss_cli = FaissManager()
    face_align_model = FaceAlign()
    face_detection_model = FaceDetector(str(BASE_DIR / "dist" / "centerface_scripted.pt"))
    face_recognition_model = FaceRecognition(str(BASE_DIR / "dist" / "edge_face_s.scripted"))

    persons = os.listdir(photos_path)
    count = 0
    persons_indexes = {}
    for person in persons:
        person_path = f"{photos_path}/{person}"
        photos = os.listdir(person_path)

        person_indexes = []
        for photo in photos:
            frame = cv2.imread(f"{person_path}/{photo}")
            boxes, _, faces_landmarks = face_detection_model.predict(frame)
            if len(boxes) != 1:
                print(f"Can not find face in image: `{person_path}/{photo}` or many faces")
                continue

            face_image = face_align_model.warp_and_crop_face(frame, faces_landmarks)
            face_box_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            emb = face_recognition_model.predict(face_box_rgb)
            faiss_cli.add_vector(emb)

            person_indexes.append(count)
            count += 1

        persons_indexes[person] = person_indexes

    faiss_cli.save_index()
    with open("filter_mapping.json", "w", encoding="UTF-8") as f:
        json.dump(persons_indexes, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, dest="photos_path", default="./celeba_train")
    arguments = parser.parse_args()

    main(arguments.photos_path)
