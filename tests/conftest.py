import cv2
import pytest


# sample video for testing
@pytest.fixture(scope="module", autouse=True)
def test_video():
    return "tests/sample/test.avi"


# sample image (frame) for testing
@pytest.fixture(scope="module", autouse=True)
def test_frame():
    return cv2.imread("tests/sample/frames/frame1.jpg")


# sample metadata for testing
@pytest.fixture(scope="module", autouse=True)
def test_metadata():
    return "tests/sample/test.json"


# sample frame details for testing
@pytest.fixture(scope="module", autouse=True)
def test_frame_details():
    return {
        "face_rect": [73, 238, 568, 986],
        "lm7pt": [
            148,
            616,
            264,
            626,
            450,
            630,
            564,
            616,
            357,
            806,
            232,
            968,
            465,
            976,
        ],
        "annot": {
            "occlusion": "TG",
            "eye_open_left": 0.9997702836990356,
            "eye_open_right": 0.9998661875724792,
        },
        "feedback": 0,
        "review_status": 1,
    }
