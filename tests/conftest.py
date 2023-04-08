import json
from typing import Any

import cv2
import pytest
from numpy import dtype, generic, ndarray


# sample image (frame) for testing
@pytest.fixture(scope="module")
def test_frame():
    """Test frame (image)."""
    return cv2.imread("tests/sample/test.jpg")


# sample frame details for testing
@pytest.fixture(scope="module")
def test_meta():
    """"""
    return json.load(open("tests/sample/test.json"))


@pytest.fixture(scope="module")
def test_sample(test_frame: ndarray[int, dtype[generic]], test_meta: Any):
    """Our format for metadata"""
    test_meta["face_landmark"] = test_meta.pop("lm7pt")
    return {"image": test_frame, "meta": test_meta}


@pytest.fixture(scope="module")
def train_dest():
    return "/Users/motorbreath/mipt/thesis/code/spoof/data/train/"


@pytest.fixture(scope="module")
def test_dest():
    return "/Users/motorbreath/mipt/thesis/code/spoof/data/test/"


@pytest.fixture(scope="module")
def train_src():
    return "/Users/motorbreath/mipt/thesis/datasets/casia/train/data/train"


@pytest.fixture(scope="module")
def test_src():
    return "/Users/motorbreath/mipt/thesis/datasets/casia/test/data/test"


@pytest.fixture(scope="module")
def train_metadata_src():
    return "/Users/motorbreath/mipt/thesis/datasets/casia/train/meta/train"


@pytest.fixture(scope="module")
def test_metadata_src():
    return "/Users/motorbreath/mipt/thesis/datasets/casia/test/meta/test"
