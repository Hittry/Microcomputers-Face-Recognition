from math import sqrt

import numpy as np

import faiss
from numpy.typing import NDArray

from const import COUNT_VECTORS_FOR_RETURN


class FaissManager:

    def __init__(self, index_path: str) -> None:
        self.index = self._prepare_index(index_path)

    @staticmethod
    def _prepare_index(index_path: str) -> faiss.IndexHNSWFlat:
        """Метод для подготовки индекса Faiss."""

        index = faiss.read_index(index_path)
        return index

    def search_sim_vector_index(self, vector: NDArray[NDArray[np.float32]], threshold: float) -> int | None:
        """Метод для получения дескриптора в Faiss."""

        distances, faiss_indexes = self.index.search(vector, k=COUNT_VECTORS_FOR_RETURN)
        min_dist = sqrt(distances[0][0])
        if min_dist <= threshold:
            return faiss_indexes[0][0]
