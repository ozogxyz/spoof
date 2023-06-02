import pytest
import pandas as pd

from spoof.dataset.dataset import FaceDataset


def test_loo_protocol():
    train_annotations_file = "data/siwm/train_annotations.csv"
    test_annotations_file = "data/siwm/test_annotations.csv"

    train_annotations = pd.read_csv(train_annotations_file)
    test_annotations = pd.read_csv(test_annotations_file)

    spoof_types = train_annotations[train_annotations["spoof_type"] != "live"][
        "spoof_type"
    ].unique()
    train_length = len(train_annotations)
    test_length = len(test_annotations)

    print(f"Initial Train Length: {train_length}")
    print(f"Initial Test Length: {test_length}")

    live_train_length = len(
        train_annotations[train_annotations["spoof_type"] == "live"]
    )
    live_test_length = len(
        test_annotations[test_annotations["spoof_type"] == "live"]
    )
    spoof_length = len(
        train_annotations[train_annotations["spoof_type"] != "live"]
    )

    print(f"Initial Live Train Length: {live_train_length}")
    print(f"Initial Live Test Length: {live_test_length}")
    print(f"Initial Spoof Length: {spoof_length}")

    for spoof_type in spoof_types:
        train_dataset = FaceDataset(annotations_file=train_annotations_file)
        train_dataset.leave_out(spoof_type)
        train_new_length = len(train_dataset.annotations)

        test_dataset = FaceDataset(annotations_file=test_annotations_file)
        test_dataset.leave_out_all_except(spoof_type)
        test_new_length = len(test_dataset.annotations)

        spoof_type_length_train = len(
            train_annotations[train_annotations["spoof_type"] == spoof_type]
        )
        spoof_type_length_test = len(
            test_annotations[test_annotations["spoof_type"] == spoof_type]
        )

        print(
            f"Train Length after leaving out {spoof_type}: {train_new_length}"
        )
        print(f"{spoof_type} Train Length: {spoof_type_length_train}")
        print(
            f"Test Length after leaving out all except {spoof_type}: {test_new_length}"
        )
        print(f"{spoof_type} Test Length: {spoof_type_length_test}")
        print()

        assert (
            train_new_length
            == live_train_length + spoof_length - spoof_type_length_train
        )
        assert test_new_length == live_test_length + spoof_type_length_test
        assert (
            train_length + test_length - spoof_length
            == train_new_length + test_new_length
        )

    print("Leave-One-Out Protocol test passed!")


if __name__ == "__main__":
    test_loo_protocol()
