import os
from typing import Literal

import pytest

from src.dataset.capture_frames import capture_frames
from src.utils import filter_files_by_ext


@pytest.mark.skip(reason="Creates zillions of files")
def test_capture(test_video: Literal['tests/sample/test.avi']):
    dest = "sample/frames"
    frame_chk = "sample/frames/frame1.jpg"

    capture_frames(src=test_video, dest=dest)

    assert os.path.exists(frame_chk)


@pytest.mark.parametrize("ext", [".avi", ".json"])
def test_filter_files_by_ext(ext: Literal[".avi", ".json"]):
    path = "sample/"
    videos = filter_files_by_ext(path, ext)

    assert all(filter(lambda x: x.endswith(ext), videos))  # type: ignore
