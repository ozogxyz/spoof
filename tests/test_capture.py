import os
from typing import Literal

import pytest

from src.dataset.capture_frames import capture_frames, extract_frames


# sample video for testing
@pytest.fixture(scope="module", autouse=True)
def test_video():
    return "tests/sample/test.avi"


def test_capture(test_video: Literal['tests/sample/test.avi']):
    dest = "sample/frames"
    frame_chk = "sample/frames/frame1.jpg"

    capture_frames(src=test_video, dest=dest)

    assert os.path.exists(frame_chk)


def test_extract_frames(test_video: Literal['tests/sample/test.avi']):
    src = "tests/sample"
    dest = "sample/frames"
    ext = ".avi"

    extract_frames(src=src, dest=dest, ext=ext)

    assert os.path.exists(dest)
