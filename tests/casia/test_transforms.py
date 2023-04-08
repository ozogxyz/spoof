from typing import Any

import cv2
import numpy as np
import pytest

from src.dataset.create_sample import CreateSample
from src.dataset.transforms import (
    FaceRegionRCXT,
    FaceRegionXT,
    em_angle,
    lm_angle,
)
import src.dataset.transforms as transforms

from src.dataset.visualize import (
    draw_face_rectangle,
    draw_landmark_points,
    show_frame,
)


@pytest.fixture(scope="module", autouse=True)
def face_region_xt():
    return FaceRegionXT(
        size=(726, 448),
        scale_rect_hw=(1, 1),
        crop=None,
        interpolation=cv2.INTER_LINEAR,
        p=0.0,
        scale_delta=0.0,
        square=False,
    )


@pytest.fixture(scope="module", autouse=True)
def face_region_rcxt():
    return FaceRegionRCXT(
        size=(726, 448),
        scale_rect_hw=(1, 1),
        crop=None,
        interpolation=cv2.INTER_LINEAR,
        p=0.0,
        scale_delta=0.0,
        square=False,
    )


# @pytest.mark.skip(reason="visualize")
def test_face_region_xt(
    face_region_xt: FaceRegionXT,
    test_sample: dict[str, Any],
):
    meta = test_sample.get("meta")
    frame = test_sample.get("image")
    sampler = CreateSample()
    sample = sampler.create_sample(meta, frame)

    new_sample = face_region_xt(sample)
    new_image, new_meta = new_sample["image"], new_sample["meta"]

    show_frame(new_image, "face_region_xt")
    assert new_image.shape[:2] == face_region_xt.size


# @pytest.mark.skip(reason="visualize")
def test_frontalize(
    face_region_rcxt: FaceRegionRCXT, test_sample: dict[str, Any]
):
    meta = test_sample.get("meta")
    frame = test_sample.get("image")
    sampler = CreateSample()
    sample = sampler.create_sample(meta, frame)

    transformed_sample = face_region_rcxt(sample)

    new_image, new_meta = (
        transformed_sample["image"],
        transformed_sample["meta"],
    )

    draw_face_rectangle(new_image, new_meta["face_rect"])
    landmark_points = new_meta["face_landmark"].flatten().tolist()
    for x, y in list(zip(landmark_points[0::2], landmark_points[1::2])):
        x, y = int(x), int(y)
        cv2.circle(new_image, (x, y), 10, (0, 0, 255), -1)
    show_frame(new_image, "face_region_rcxt")

    print(lm_angle(new_meta["face_landmark"]))
    print(em_angle(new_meta["face_landmark"]))
    assert new_image.shape[:2] == face_region_rcxt.size


def test_transforms(test_sample: dict[str, Any]):
    lms = np.array(test_sample["meta"]["face_landmark"]).reshape(7, 2)
    test_sample["meta"]["face_landmark"] = lms
    frontalize = transforms.FaceRegionRCXT(size=(448, 448))

    # test frontalize
    eps_lm = 1e-6
    eps_em = 5
    frontalized = frontalize(test_sample)
    assert frontalized["image"].shape[:2] == frontalize.size
    assert frontalized["meta"]["face_landmark"].shape == (7, 2)
    assert lm_angle(frontalized["meta"]["face_landmark"]) < eps_lm
    assert em_angle(frontalized["meta"]["face_landmark"]) - 90 < eps_em


def test_transforms_with_sampler(test_sample: dict[str, Any]):
    test_meta = test_sample["meta"]
    test_frame = test_sample["image"]
    frontalize = transforms.FaceRegionRCXT(size=(448, 448))

    sampler = CreateSample()
    sample = sampler.create_sample(test_meta, test_frame)

    # test frontalize
    eps_lm = 1e-6
    eps_em = 5
    frontalized = frontalize(sample)
    assert frontalized["image"].shape[:2] == frontalize.size
    assert frontalized["meta"]["face_landmark"].shape == (7, 2)
    assert lm_angle(frontalized["meta"]["face_landmark"]) < eps_lm
    assert em_angle(frontalized["meta"]["face_landmark"]) - 90 < eps_em
