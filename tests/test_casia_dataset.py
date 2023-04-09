import os
from pathlib import Path

import pytest

from src.preprocessing.casia import extract_frames, extract_metadata


@pytest.mark.skip(reason="This test takes a long time to run")
def test_extract_train_frames():
    video_src = "data/casia/train/data/train"
    save_dest = "data/casia/train_frames"
    fr_n = extract_frames(video_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 20

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(save_dest).rglob("*.jpg"):
        count += 1
    assert fr_n == count


@pytest.mark.skip(reason="This test takes a long time to run")
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


# @pytest.mark.skip(reason="This test takes a long time to run")
def test_extract_train_meta():
    meta_src = "data/casia/train/meta/train"
    save_dest = "data/casia/train_frames"
    mn = extract_metadata(meta_src, save_dest)

    # Check that the number of folders extracted is correct
    assert len(os.listdir(save_dest)) == 20

    # Check that the number of frames extracted is correct
    count = 0
    for path in Path(save_dest).rglob("*.json"):
        count += 1
    assert mn == count


@pytest.mark.skip(reason="This test takes a long time to run")
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
