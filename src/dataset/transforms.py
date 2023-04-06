from copy import deepcopy
import random
from typing import Dict, List, Tuple
import warnings

import cv2
import numpy as np


# get face rectangle from landmarks
def rect_from_lm(landmarks: List[int], scale: int = 1) -> List[int]:
    """
    Get face rectangle from landmarks.

    Args:
        landmarks (List[int]): list of 7 point landmarks
        scale (int): scale factor

    Returns:
        List[int]: face rectangle
    """

    lms = np.array(landmarks).ravel()
    if len(lms) == 14:
        lms = lms.reshape(7, -1)
    assert lms.shape == (7, 2), "wrong lms dimensions: {}".format(lms.shape)
    eye1 = lms[:2].mean(axis=0)
    eye2 = lms[2:4].mean(axis=0)
    mouth = lms[5:].mean(axis=0)
    face_center = (((eye1 + eye2) / 2 + mouth) / 2).astype(int)
    eye_dist = np.linalg.norm(eye1 - eye2)
    eyes_to_mouth = np.linalg.norm((eye1 + eye2) / 2 - mouth)
    fw = eye_dist * 2
    fh = eyes_to_mouth * 1.7
    fsize = int(max(fw * scale, fh * scale))
    face_rect = [
        max(0, face_center[0] - fsize // 2),
        max(0, face_center[1] - fsize // 2),
        fsize,
        fsize,
    ]
    return np.int32(face_rect).ravel().tolist()


def lm_angle(lms: np.ndarray) -> float:
    """Calculate angle between eyes and x-axis.

    Args:

        lms (np.ndarray): array of shape [7, 2] or [5, 2] with
                          key point coodrinates in [X, Y] format.
    """
    if lms.shape[0] == 7:
        eye_l = lms[:2].mean(axis=0)
        eye_r = lms[2:4].mean(axis=0)
    elif lms.shape[0] == 5:
        eye_l = lms[0]
        eye_r = lms[1]
    else:
        raise NotImplementedError(
            f"incorrect landmark shape: supported [5, 2] or [7, 2], got: {lms.shape}"
        )

    diff = eye_r - eye_l
    lm_angle = np.arctan2(diff[1], diff[0]) / np.pi * 180
    return lm_angle


def em_angle(lms: np.ndarray) -> float:
    if lms.shape[0] == 7:
        eye_center = (lms[:2].mean(axis=0) + lms[2:4].mean(axis=0)) / 2
    elif (
        lms.shape[0] == 5
    ):  # eye_left , eye_right, nose, mouth_left, mouth_right
        eye_center = lms[:2].mean(axis=0)
    else:
        raise NotImplementedError(
            f"incorrect landmark shape: supported [5, 2] or [7, 2], got: {lms.shape}"
        )
    mouth_center = lms[-2:].mean(axis=0)
    em_line = mouth_center - eye_center
    angle = np.arctan2(em_line[1], em_line[0]) / np.pi * 180
    return angle


def _get_size_value(size_param):
    if size_param is None:
        return size_param
    elif isinstance(size_param, int) and size_param > 0:
        assert (
            size_param > 0
        ), f"Incorrect input: size should be > 0, got {size_param}"
        return size_param, size_param
    elif (
        isinstance(size_param, tuple) or isinstance(size_param, list)
    ) and len(size_param) == 2:
        assert isinstance(
            size_param[0], int
        ), f"Incorrect size[0] type, int expected, got {type(size_param[0])}"
        assert isinstance(
            size_param[1], int
        ), f"Incorrect size[1] type, int expected, got {type(size_param[1])}"
        assert (
            size_param[0] > 0
        ), f"Incorrect input: size[0] should be > 0, got {size_param[0]}"
        assert (
            size_param[1] > 0
        ), f"Incorrect input: size[1] should be > 0, got {size_param[1]}"
        return size_param
    else:
        raise ValueError(f"'size' parameter not understood: {size_param}")


"""
All the ROI transformation classes expects input in format of sample dict with specific keys.
Output is also in sample dictionary format.

sample dictionary description:
{
    "image": 3-channel BGR image, np.ndarray of shape [H, W, 3],
    "meta": {
        
        "face_rect": list or np.ndarray for face rectangle
            in format [xx, yy, ww, hh]. (xx, yy) - coordinate of left top angle of rectangle, ww and hh - its width and height
        
        "face_landmark": np.narray of shape [7, 2] or [5, 2] with  key point coodrinates in [X, Y] format.
            - point order for 7-point version:
                 [
                    left corner of left eye,
                    right corner of left eye,
                    left corner of right eye,
                    right corner of right eye,
                    nose point,
                    mouth left corner point,
                    mouth right corner point
                 ]
            - point order for 5-point version:
                 [
                    left eye center,
                    right eye center,
                    nose point,
                    mouth left corner point,
                    mouth right corner point
                 ]
    }
}
"""

## ROI EXTRACTION / TRANSFORMATION classes
class ScalingTransform:
    def __init__(self, size: Tuple[int, int] | None = None):
        self.size = _get_size_value(size)


class MetaAddLMRect:
    """replace the face rectangle with"""

    def __init__(self, scale_f: Tuple[int, int] = (1, 1)):
        self.scale_f = scale_f

    def __call__(self, sample: Dict) -> Dict:
        meta = deepcopy(sample["meta"])
        lm_array = meta["face_landmark"]
        lm_type = meta.get("face_landmark_type", "LM7")
        if lm_type == "LM7":
            lm_rect = rect_from_lm(lm_array)
            meta["face_rect"] = np.array(lm_rect)
        else:
            warnings.warn(f"Landmark type not supported: {lm_type}")

        sample["meta"] = meta
        return sample


class MetaRandomNoise:
    """For training augmentation: add noise to face keypoint(landmark) positions"""

    def __init__(self, p=0.5, max_shift: Tuple[float, float] = (0.0, 0.0)):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, sample: Dict) -> Dict:
        if random.random() > self.p:
            return sample

        meta = deepcopy(sample["meta"])

        lm_array = np.float32(meta["face_landmark"])
        lm_size = np.std(lm_array, axis=0)
        lm_shift_size = lm_size * np.array(self.max_shift)
        lm_noise_x = np.array(
            [
                np.float32((2 * random.random() - 1) * lm_shift_size[0])
                for _ in lm_array
            ]
        )
        lm_array[:, 0] += lm_noise_x

        lm_noise_y = np.array(
            [
                np.float32((2 * random.random() - 1) * lm_shift_size[1])
                for _ in lm_array
            ]
        )
        lm_array[:, 1] += lm_noise_y

        meta["face_landmark"] = lm_array
        sample["meta"] = meta
        return sample


class FaceRegionXT(ScalingTransform):
    """
    Extract face rectangle area with rectangle scaled N(float) times.
    Can either resize face rectangle area or crop it if 'crop' parameter is not None
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        scale_rect_hw: Tuple[int, int] = (1, 1),
        crop: Tuple[int, int] = None,
        interpolation=cv2.INTER_LINEAR,
        p=0.0,
        scale_delta=0.0,
        square=False,
    ):
        super().__init__(size=size)
        self.interpolation = interpolation
        self.crop = _get_size_value(crop)
        self.p = p
        self.scale_delta = scale_delta
        self.square = square

        assert (
            scale_rect_hw[1] > 0
        ), f"scale factor for X should be greater than 0, got {scale_rect_hw[1]:.2f}"
        assert (
            scale_rect_hw[0] > 0
        ), f"scale factor for Y should be greater than 0, got {scale_rect_hw[0]:.2f}"
        self.scale_rect_hw = scale_rect_hw

    def get_affine_transform(
        self, meta: Dict
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        rect = meta["face_rect"]

        rx, ry, rw, rh = rect
        rcx = rx + rw / 2
        rcy = ry + rh / 2
        if self.square:
            max_rhw = max(rh, rw)
            rw = rh = max_rhw

        scale_h, scale_w = self.scale_rect_hw

        # random scale for augmetation during training
        if random.random() < self.p:
            scale_h = scale_h + self.scale_delta * (2 * random.random() - 1)
            scale_w = scale_w + self.scale_delta * (2 * random.random() - 1)

        rw = rw * scale_w
        rh = rh * scale_h

        size_x = rw if self.crop is None else self.crop[0]
        size_y = rh if self.crop is None else self.crop[1]

        if self.crop is None and self.size is not None:
            scale_x = self.size[0] / size_x
            scale_y = self.size[1] / size_y
            dsize = self.size
        else:
            scale_x, scale_y = 1, 1
            dsize = (size_x, size_y)

        offset_mat = np.array(
            [[1, 0, -rcx], [0, 1, -rcy], [0, 0, 1]], dtype=np.float32
        )
        reset_mat = np.array(
            [[1, 0, dsize[0] / 2], [0, 1, dsize[1] / 2], [0, 0, 1]],
            dtype=np.float32,
        )

        scale_mat = np.array(
            [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32
        )
        affine_mat = reset_mat @ (scale_mat @ offset_mat)
        return affine_mat[:2, :], dsize

    def update_meta(self, meta: Dict, affine_mat: np.ndarray) -> Dict:
        new_meta = {k: v for k, v in meta.items()}

        lm_array = meta["face_landmark"]
        lm_dst = lm_array @ affine_mat[:2, :2].T + affine_mat[:2, -1:].T
        new_meta["face_landmark"] = lm_dst

        rect_array = np.array(meta["face_rect"])
        rect_array[:2] = (
            rect_array[None, :2] @ affine_mat[:2, :2].T + affine_mat[:2, -1:].T
        )
        rect_array[-2:] = rect_array[None, -2:] @ affine_mat[:2, :2].T
        new_meta["face_rect"] = rect_array.tolist()

        return new_meta

    def __call__(self, sample: Dict) -> Dict:
        meta = sample["meta"]

        # get rect cropping matrix
        affine_mat, dst_size = self.get_affine_transform(meta)

        # update meta data
        meta_dst = self.update_meta(meta, affine_mat)

        # crop image
        dst_image = cv2.warpAffine(
            sample["image"].copy(),
            affine_mat,
            dst_size,
            flags=self.interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sample_dst = deepcopy(sample)
        sample_dst["image"] = dst_image
        sample_dst["meta"] = meta_dst
        return sample_dst


class FaceRegionRCXT(FaceRegionXT):
    """Cropping face area with rotation compensation based on nose line (line from center of eye positions to center of mouth edges)."""

    def get_affine_transform(
        self, meta: Dict
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        rect = meta["face_rect"]
        lm_array = meta["face_landmark"]
        angle = (em_angle(lm_array) - 90) / 180 * np.pi

        rx, ry, rw, rh = rect
        rcx = rx + rw / 2
        rcy = ry + rh / 2

        scale_h, scale_w = self.scale_rect_hw
        # random scale
        if random.random() < self.p:
            scale_h = scale_h + self.scale_delta * (2 * random.random() - 1)
            scale_w = scale_w + self.scale_delta * (2 * random.random() - 1)
            # print(f"sampled scale HxW: {scale_h:.2f} x {scale_w:.2f}")
        rw = rw * scale_w
        rh = rh * scale_h

        size_x = rw if self.crop is None else self.crop[0]
        size_y = rh if self.crop is None else self.crop[1]

        if self.crop is None:
            scale_x = self.size[0] / size_x
            scale_y = self.size[1] / size_y
            dsize = self.size
        else:
            scale_x, scale_y = 1, 1
            dsize = (size_x, size_y)

        offset_mat = np.array(
            [[1, 0, -rcx], [0, 1, -rcy], [0, 0, 1]], dtype=np.float32
        )
        reset_mat = np.array(
            [[1, 0, dsize[0] / 2], [0, 1, dsize[1] / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        rot_mat = np.array(
            [
                [scale_x * np.cos(angle), scale_x * np.sin(angle), 0],
                [-scale_y * np.sin(angle), scale_y * np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        affine_mat = reset_mat @ (rot_mat @ offset_mat)
        return affine_mat[:2, :], dsize
