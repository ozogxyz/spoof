from copy import deepcopy
from typing import Dict

import numpy as np


class CreateSample:
    def __call__(self, sample: Dict) -> Dict:
        return self.create_sample(sample["meta"], sample["image"])

    def _validate_face_rect(self, meta: Dict, key: str) -> Dict:
        """Validate if face_rect is in correct format."""
        face_rect = meta.get(key)
        assert face_rect is not None, "Face rect not found in meta"
        if isinstance(face_rect, list):
            face_rect = np.array(face_rect).ravel()
        assert face_rect.shape == (4,), "Face rect shape is wrong"
        meta[key] = face_rect

        return meta

    def _validate_landmarks(self, meta: Dict, key: str) -> Dict:
        """Validate if landmark is in correct format."""
        face_landmarks = meta.get(key)
        assert face_landmarks is not None, "Landmark not found in meta"
        if isinstance(face_landmarks, list):
            face_landmarks = np.array(face_landmarks).ravel()
        if len(face_landmarks) == 14:
            face_landmarks = face_landmarks.reshape(7, -1)
        assert face_landmarks.shape == (
            7,
            2,
        ), "Landmark shape is wrong: {}".format(face_landmarks.shape)

        meta[key] = face_landmarks

        return meta

    def _rename_keys(self, meta: Dict, new_key: str, old_key: str) -> Dict:
        if old_key in meta.keys():
            meta = deepcopy(meta)
            meta[new_key] = meta.pop(old_key)

        return meta

    def create_sample(self, meta: Dict, frame) -> Dict:
        """Add metadata in our format to a frame np array.
        For now only adheres to CASIA format, ideally should convert
        any metadata to our format. In CASIA keys are face_rect
        and lm7pt.
        Args:
            meta: metadata of a frame obtained from data
            frame: np array of a frame
        Returns:
            sample: a dict with keys "image" and "meta"
        """
        meta = self._rename_keys(meta, "face_landmark", "lm7pt")
        meta = self._validate_face_rect(meta, "face_rect")
        meta = self._validate_landmarks(meta, "face_landmark")

        sample = {
            "image": frame,
            "meta": meta,
        }

        return sample
