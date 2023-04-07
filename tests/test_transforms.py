from typing import Any

import cv2
import pytest

from src.dataset.add_metadata import add_metadata
from src.dataset.transforms import (
    FaceRegionRCXT,
    FaceRegionXT,
    em_angle,
    lm_angle,
)
from src.dataset.visualize import (
    draw_face_rectangle,
    draw_landmark_points,
    show_frame,
)


@pytest.fixture(scope="module", autouse=True)
def face_region_xt():
    return FaceRegionXT(
        size=(448, 726),
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


@pytest.mark.skip(reason="visualize")
def test_face_region_xt(
    face_region_xt: FaceRegionXT,
    test_sample: dict[str, Any],
):
    meta = test_sample.get("meta")
    frame = test_sample.get("image")
    sample = add_metadata(meta, frame)  # type: ignore

    new_sample = face_region_xt(sample)
    new_image, new_meta = new_sample["image"], new_sample["meta"]

    show_frame(new_image, "face_region_xt")
    # assert new_image.shape[:2] == face_region_xt.size


# @pytest.mark.skip(reason="visualize")
def test_frontalize(
    face_region_rcxt: FaceRegionRCXT, test_sample: dict[str, Any]
):
    meta = test_sample.get("meta")
    frame = test_sample.get("image")
    sample = add_metadata(meta, frame)  # type: ignore

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
