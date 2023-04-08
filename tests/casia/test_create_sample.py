import pytest

from src.dataset.create_sample import CreateSample


def test_create_sample(test_meta, test_frame):
    sampler = CreateSample()

    sample = sampler.create_sample(test_meta, test_frame)

    assert sample["meta"]["face_landmark"].shape == (7, 2)
    assert sample["image"].shape == (
        test_frame.shape[0],
        test_frame.shape[1],
        3,
    )


def test_create_sample_wrong_dimensions(test_meta, test_frame):
    with pytest.raises(AssertionError):
        sampler = CreateSample()

        test_meta["lm7pt"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sampler.create_sample(test_meta, test_frame)
