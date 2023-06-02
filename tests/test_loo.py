import pytest
from spoof.dataset.dataset import FaceDataset


def test_leave_out():
    train_annotations_file = "data/siwm/train_annotations.csv"
    train_dataset = FaceDataset(train_annotations_file)
    spoof_types = train_dataset.annotations["spoof_type"].unique()
    initial_length = len(train_dataset)

    for spoof_type in spoof_types:
        train_dataset_leave_out = FaceDataset(train_annotations_file)
        train_dataset_leave_out.leave_out(spoof_type)
        new_length = len(train_dataset_leave_out)

        expected_removed_samples = (
            train_dataset.annotations["spoof_type"]
            .value_counts()
            .get(spoof_type, 0)
        )

        assert new_length == initial_length - expected_removed_samples
        assert (
            spoof_type
            not in train_dataset_leave_out.annotations["spoof_type"].unique()
        )


def test_leave_out_all_except():
    test_annotations_file = "data/siwm/test_annotations.csv"
    test_dataset = FaceDataset(test_annotations_file)
    spoof_types = test_dataset.annotations["spoof_type"].unique()

    for spoof_type in spoof_types:
        test_dataset_leave_out_all_except = FaceDataset(test_annotations_file)
        test_dataset_leave_out_all_except.leave_out_all_except(spoof_type)
        new_length = len(test_dataset_leave_out_all_except)

        expected_removed_samples = (
            test_dataset.annotations["spoof_type"]
            .value_counts()
            .get(spoof_type, 0)
        )

        assert new_length == expected_removed_samples
        assert (
            spoof_type
            in test_dataset_leave_out_all_except.annotations[
                "spoof_type"
            ].unique()
        )
        assert (
            len(
                test_dataset_leave_out_all_except.annotations[
                    "spoof_type"
                ].unique()
            )
            == 1
        )
