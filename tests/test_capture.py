import os
from typing import Literal

import pytest

from src.dataset.capture_frames import capture_frames
from src.utils import filter_files_by_ext


@pytest.mark.skip(reason="Creates zillions of files")
def test_capture():
    src = "/Users/motorbreath/mipt/thesis/datasets/casia/casia-mfsd_train_renamed/data/train/001M/C001_HR_E1_IN_TG_00D_PT+HR+1_0_1.avi"
    dest = "/Users/motorbreath/mipt/thesis/code/spoof/tmp/frames"
    frame_chk = "/Users/motorbreath/mipt/thesis/code/spoof/tmp/frame1.jpg"

    capture_frames(src=src, dest=dest)

    assert os.path.exists(frame_chk)


@pytest.mark.parametrize("ext", [".avi", ".json"])
def test_filter_files_by_ext(ext: Literal[".avi", ".json"]):
    path = "/Users/motorbreath/mipt/thesis/datasets/casia/casia-mfsd_train_renamed/data/train/001M/"
    videos = filter_files_by_ext(path, ext)

    assert all(filter(lambda x: x.endswith(ext), videos))
