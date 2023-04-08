import os

import pytest

from src.dataset.extract_frames import extract_frames, create_labels_csv


# @pytest.skip(reason="Too long to run")
# def test_extract_frames():
#     train_src = (
#         "/Users/motorbreath/mipt/thesis/datasets/casia/train/data/train"
#     )
#     dest = "tests/sample/frames"
#     ext = ".avi"

#     extract_frames(src=train_src, dest=dest, ext=ext)

#     assert os.path.exists(dest)
#     assert os.path.exists(os.path.join(dest, "labels.csv"))


def test_create_labels_csv():
    frame_src = "/Users/motorbreath/mipt/thesis/code/spoof/tests/sample/frames"
    dest = "tests/sample/"

    create_labels_csv(frame_src, dest=dest)
    assert os.path.exists(os.path.join(dest, "labels.csv"))
