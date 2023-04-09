import os
from pathlib import Path
import cv2
import numpy as np

import pytest

from src.preprocessing.casia import Sampler, extract_frames, extract_metadata, CASIA
from src.preprocessing.transforms import FaceRegionRCXT
from src.utils.annotations import create_annotations
from src.utils.visualize import show_frame


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
    transform = FaceRegionRCXT(size=(224, 224))
    dataset = CASIA(
        "data/casia/train_frames",
        annotations="data/casia/annotations.txt",
        meta_file="data/casia/meta.txt",
        transform=transform,
        sampler=Sampler(),
    )

    assert len(dataset) == 45141
    assert len(open("data/casia/annotations.txt").readlines()) == 45141
    assert len(open("data/casia/meta.txt").readlines()) == 45141

    # convert tensor to opencv image
    # image = dataset[1241][0].numpy().transpose(1, 2, 0)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # show_frame(title="Sample input to CNN", frame=image)
    # assert len(dataset) == 45141
    # assert image.shape == (224, 224, 3)
