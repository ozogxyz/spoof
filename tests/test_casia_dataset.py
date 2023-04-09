import os
from pathlib import Path

from src.preprocessing.casia import extract_metadata


def test_extract_train_frames():
    video_src = "data/casia/train"
    save_dest = "data/casia/train_frames"
    # extract_frames(video_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 20

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(save_dest).rglob("*.jpg"):
        count += 1
    assert count == 45141


def test_extract_test_frames():
    video_src = "data/casia/test"
    save_dest = "data/casia/test_frames"
    # extract_frames(video_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 30

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(save_dest).rglob("*.jpg"):
        count += 1
    assert count == 65888


def test_extract_train_meta():
    meta_src = "data/casia/train/meta/train"
    save_dest = "data/casia/train/train_frames"
    extract_metadata(meta_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 20

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(meta_src).rglob("*.json"):
        count += 1
    assert count == 20
