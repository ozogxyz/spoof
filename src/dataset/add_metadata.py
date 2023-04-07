from copy import deepcopy
from typing import Dict

import numpy as np


def ensure_nparray_lm7(meta: Dict, key: str = "face_landmark") -> Dict:
    """Validate if meta["key"] to a 7 pt landmark np array"""
    face_landmarks = meta.get(key)
    if isinstance(face_landmarks, list):
        meta[key] = np.asarray(face_landmarks, dtype=int).reshape(7, -1)

    return meta


def ensure_nparray_face_rect(meta: Dict, key: str = "face_rect") -> Dict:
    face_rect = meta.get(key)
    if isinstance(face_rect, list):
        meta[key] = np.asarray(face_rect, dtype=int)

    return meta


def rename_keys(meta: Dict, new_key: str, old_key: str) -> Dict:
    if old_key in meta.keys():
        meta = deepcopy(meta)
        meta[new_key] = meta.pop(old_key)

    return meta


def add_metadata(meta: Dict, frame) -> Dict:
    """Add metadata in our format to a frame np array.
    For now only adheres to CASIA format, ideally should convert
    any metadata to our format.
    Args:
        meta: metadata of a frame obtained from data supplier
        frame: cv2 frame read from captured images

    Returns:
        sample: new dict of metadata in our format
    """
    meta = (
        rename_keys(meta, "face_landmark", "lm7pt")
        if "face_landmark" not in meta.keys()
        else meta
    )
    meta = ensure_nparray_face_rect(meta)
    meta = ensure_nparray_lm7(meta)

    sample = dict()
    sample["image"] = frame
    sample["meta"] = meta

    return sample
