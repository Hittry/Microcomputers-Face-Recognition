import numpy as np

import faiss
from numpy.typing import NDArray

from const import (
    BASE_DIR,
    FAISS_INDEX_NAME,
    FAISS_M,
    FAISS_EF_SEARCH,
    FAISS_EF_CONSTRUCTION,
    FAISS_VECTOR_DIM,
)


class FaissManager:

    def __init__(self) -> None:
        self.index = self._prepare_index()

    @staticmethod
    def _prepare_index() -> faiss.IndexHNSWFlat:
        """Метод для подготовки индекса Faiss."""

        index = faiss.IndexHNSWFlat(FAISS_VECTOR_DIM, FAISS_M)
        index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
        index.hnsw.efSearch = FAISS_EF_SEARCH
        return index

    def add_vector(self, vector: NDArray[NDArray[np.float32]]) -> None:
        """Метод для добавления дескриптора в Faiss."""

        self.index.add(vector)

    def save_index(self) -> None:
        """Метод для сохранения индекса."""

        faiss.write_index(self.index, str(BASE_DIR / FAISS_INDEX_NAME))
