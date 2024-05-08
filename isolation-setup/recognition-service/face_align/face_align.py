import cv2
import numpy as np
from numpy.typing import NDArray

from const import REFERENCE_FACE_POINTS, FACE_CROP_SIZE


class FaceAlign:

    def __init__(self) -> None:
        self.ref_face_points = REFERENCE_FACE_POINTS
        self.crop_size = FACE_CROP_SIZE

    @staticmethod
    def _tformfwd(trans, uv):
        uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
        xy = np.dot(uv, trans)
        xy = xy[:, 0:-1]
        return xy

    @staticmethod
    def _find_nonreflective_similarity(uv, xy):

        K = 2
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))
        y = xy[:, 1].reshape((-1, 1))

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))

        u = uv[:, 0].reshape((-1, 1))
        v = uv[:, 1].reshape((-1, 1))
        U = np.vstack((u, v))

        if np.linalg.matrix_rank(X) >= 2 * K:
            r, _, _, _ = np.linalg.lstsq(X, U)
            r = np.squeeze(r)
        else:
            raise Exception

        sc, ss, tx, ty = r[0], r[1], r[2], r[3]
        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])
        T = np.linalg.inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])
        return T, Tinv

    def _find_similarity(self, uv, xy):
        trans1, trans1_inv = self._find_nonreflective_similarity(uv, xy)
        xyR = xy
        xyR[:, 0] = -1 * xyR[:, 0]

        TreflectY = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        trans2r, trans2r_inv = self._find_nonreflective_similarity(uv, xyR)
        xy1 = self._tformfwd(trans1, uv)
        norm1 = np.linalg.norm(xy1 - xy)

        trans2 = np.dot(trans2r, TreflectY)
        xy2 = self._tformfwd(trans2, uv)
        norm2 = np.linalg.norm(xy2 - xy)

        if norm1 <= norm2:
            return trans1, trans1_inv
        else:
            trans2_inv = np.linalg.inv(trans2)
            return trans2, trans2_inv

    def _get_similarity_transform(self, src_pts, dst_pts, reflective=True):
        if reflective:
            trans, trans_inv = self._find_similarity(src_pts, dst_pts)
        else:
            trans, trans_inv = self._find_nonreflective_similarity(src_pts, dst_pts)
        return trans, trans_inv

    def _get_similarity_transform_for_cv2(self, src_pts, dst_pts, reflective=True):
        trans, trans_inv = self._get_similarity_transform(src_pts, dst_pts, reflective)
        cv2_trans = trans[:, 0:2].T
        return cv2_trans

    def warp_and_crop_face(self, src_img: NDArray, facial_pts: NDArray) -> NDArray:
        ref_pts = np.float32(self.ref_face_points)
        ref_pts_shp = ref_pts.shape
        if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
            raise Exception

        if ref_pts_shp[0] == 2:
            ref_pts = ref_pts.T

        src_pts = np.float32(facial_pts)
        src_pts_shp = src_pts.shape
        if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
            raise Exception

        if src_pts_shp[0] == 2:
            src_pts = src_pts.T

        if src_pts.shape != ref_pts.shape:
            raise Exception

        tfm = self._get_similarity_transform_for_cv2(src_pts, ref_pts)
        face_img = cv2.warpAffine(src_img, tfm, self.crop_size)
        return face_img
