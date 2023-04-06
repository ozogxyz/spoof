from typing import Dict

import numpy as np
import pytest
from src.dataset.transforms import (
    _get_size_value,
    em_angle,
    lm_angle,
    rect_from_lm,
    ScalingTransform,
)


# @pytest.mark.skip(reason="What should this test?")
def test_rect_from_lm(test_landmarks, test_face_square):
    face_rect = rect_from_lm(test_landmarks)

    assert face_rect == test_face_square


def test_lm_angle(test_landmarks):
    landmarks = np.asarray(test_landmarks).reshape(-1, 2)
    angle = lm_angle(landmarks)

    assert angle == 0


def test_em_angle(test_landmarks):
    landmarks = np.asarray(test_landmarks).reshape(-1, 2)
    angle = em_angle(landmarks)

    assert angle == 0


def test_scaling_transform_none(size_param=None):
    scaling_transform = ScalingTransform(size_param)
    assert scaling_transform.size == size_param


def test_scaling_transform_int(size_param=256):
    scaling_transform = ScalingTransform(size_param)
    assert scaling_transform.size == (size_param, size_param)


def test_scaling_transform_tuple(size_param=(256, 256)):
    scaling_transform = ScalingTransform(size_param)
    assert scaling_transform.size == size_param


def test_scaling_transform_list(size_param=[256, 256]):
    scaling_transform = ScalingTransform(size_param)
    assert scaling_transform.size == list(size_param)


# Invalid input sizes & types
def test_scaling_transform_int_fail():
    with pytest.raises(ValueError):
        ScalingTransform("some_string")


def test_scaling_transform_tuple_fail():
    with pytest.raises(ValueError):
        ScalingTransform((256, 256, 256))


def test_scaling_transform_list_fail():
    with pytest.raises(ValueError):
        ScalingTransform([256, 256, 256])
