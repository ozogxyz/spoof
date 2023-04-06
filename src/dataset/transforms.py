from copy import deepcopy
import random
from typing import Dict, List, Tuple
import warnings

import cv2
import numpy as np


# get face rectangle from landmarks
def get_face_rect(landmarks: List[int], scale: int = 1) -> List[int]:
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
