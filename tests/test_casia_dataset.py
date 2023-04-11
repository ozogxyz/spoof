import os
from pathlib import Path
import cv2

import pytest
from torchvision.transforms import Compose

from src.preprocessing.casia import extract_frames, extract_metadata, CASIA
from src.preprocessing.transforms import FaceRegionRCXT, MetaAddLMSquare, rect_from_lm

from src.preprocessing.casia2 import CASIA2
from src.utils.visualize import show_frame, draw_face_rectangle, draw_landmarks


@pytest.mark.skip(reason="Slow")
def test_extract_train_frames():
    video_src = "data/casia/train/data/train"
    save_dest = "data/casia/train_frames"
    fr_n = extract_frames(video_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 20

    # Check that the number of frames extracted is correct
    count = 0
    for _ in Path(save_dest).rglob("*.jpg"):
        count += 1
    assert count == 45141
    assert fr_n == count
    assert fr_n == 45141


@pytest.mark.skip(reason="Slow")
def test_extract_test_frames():
    video_src = "data/casia/test"
    save_dest = "data/casia/test_frames"
    fr_n = extract_frames(video_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 30

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(save_dest).rglob("*.jpg"):
        count += 1
    assert fr_n == count


@pytest.mark.skip(reason="Slow")
def test_extract_train_meta():
    meta_src = "data/casia/train/meta/train"
    save_dest = "data/casia/train_frames"
    mn = extract_metadata(meta_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 20

    # Check that the number of frames extracted is correct
    count = 0
    for _ in Path(save_dest).rglob("*.json"):
        count += 1
    assert count == 45141
    assert mn == count
    assert mn == 45141


@pytest.mark.skip(reason="Missing folders")
def test_extract_test_meta():
    meta_src = "data/casia/test/meta/test"
    save_dest = "data/casia/test_frames"
    mn = extract_metadata(meta_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 30

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(save_dest).rglob("*.json"):
        count += 1
    assert mn == count


# @pytest.mark.skip(reason="Slow")
def test_casia_dataset():
    align = FaceRegionRCXT(size=(224, 224))
    sq = MetaAddLMSquare()
    transform = Compose([sq, align])

    dataset = CASIA2(
        annotations_path="data/casia/images/train/annotations.json",
        root_dir="data/casia/images/train",
        transform=transform,
    )

    image = dataset[0][0]["image"]
    face_rect = dataset[0][0]["meta"]["face_rect"]
    face_landmark = dataset[0][0]["meta"]["face_landmark"]

    # face_rect = rect_from_lm(face_landmark)

    print(face_rect)
    print(face_landmark)

    draw_face_rectangle(image, face_rect)
    draw_landmarks(image, face_landmark)
    # show_frame(title="IMG", frame=image)
    cv2.imwrite("tests/transformed_img.jpg", img=image)

    assert len(dataset) == 44673
    assert len(dataset[0]) == 2
    assert dataset[0][0]["image"].shape == (224, 224, 3)
    assert len(dataset[0][0]["meta"]["face_rect"]) == 4
    assert dataset[0][0]["meta"]["face_landmark"].shape == (7, 2)
