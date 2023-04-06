from typing import Any, Dict

import cv2
import pytest

from src.dataset.transforms import (
    FaceRegionRCXT,
    FaceRegionXT,
    lm_angle,
    em_angle,
    rect_from_lm,
)

from src.dataset.add_metadata import add_metadata
from src.dataset.visualize import show_frame


@pytest.fixture(scope="module", autouse=True)
def face_region_xt():
    return FaceRegionXT(
        size=(224, 224),
        scale_rect_hw=(1, 1),
        crop=None,
        interpolation=cv2.INTER_LINEAR,
        p=0.0,
        scale_delta=0.0,
        square=True,
    )


@pytest.fixture(scope="module", autouse=True)
def face_region_rcxt():
    return FaceRegionRCXT(
        size=(224, 224),
        scale_rect_hw=(1, 1),
        crop=None,
        interpolation=cv2.INTER_LINEAR,
        p=0.0,
        scale_delta=0.0,
        square=True,
    )


def test_rect_from_lm(test_sample: Dict):
    """Test that the face rectangle is correctly extracted as a square"""
    test_landmarks = test_sample["lm7pt"]
    face_rect = rect_from_lm(test_landmarks)

    assert face_rect == [51, 496, 602, 602]
    assert face_rect[2] == face_rect[3]


def test_face_region_xt(face_region_xt: FaceRegionXT, test_sample, test_frame):
    new_sample = face_region_xt(sample=add_metadata(test_sample, test_frame))
    new_image, new_meta = new_sample["image"], new_sample["meta"]

    show_frame(new_image)
    # print(new_meta)
    # assert False


def test_face_region_rxct(
    face_region_rcxt: FaceRegionRCXT, test_sample: Dict, test_frame
):
    new_sample = face_region_rcxt(sample=add_metadata(test_sample, test_frame))
    new_image, new_meta = new_sample["image"], new_sample["meta"]

    show_frame(new_image)
    # print(new_meta)
    # assert False
