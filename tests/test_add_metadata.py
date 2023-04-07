import numpy as np
from src.dataset.add_metadata import (
    ensure_nparray_face_rect,
    ensure_nparray_lm7,
    add_metadata,
)

# def test_ensure_nparray_lm7(test_sample):
#     sample = ensure_nparray_lm7(test_sample)
#     assert isinstance(sample["lm7pt"], np.ndarray)


# def test_ensure_nparray_face_rect(test_sample):
#     sample = ensure_nparray_face_rect(test_sample)
#     assert isinstance(sample["face_rect"], np.ndarray)


def test_add_metadata(test_sample, test_frame):
    new_meta = add_metadata(test_sample, test_frame)
    assert "image" in new_meta.keys()
    assert "meta" in new_meta.keys()
