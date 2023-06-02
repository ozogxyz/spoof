import pandas as pd
import os


def test_dataset_overlap():
    train_annotations_file = "data/siwm/train_annotations.csv"
    test_annotations_file = "data/siwm/test_annotations.csv"

    # Load train and test annotation files into dataframes
    train_df = pd.read_csv(train_annotations_file)
    test_df = pd.read_csv(test_annotations_file)

    # Extract filenames or identifiers from train and test dataframes
    train_filenames = set(train_df["image_path"])
    test_filenames = set(test_df["image_path"])

    # Check for overlap between train and test datasets
    overlap = train_filenames.intersection(test_filenames)
    assert (
        len(overlap) == 0
    ), "There is overlap between train and test datasets."


def test_leave_one_out():
    train_annotations_file = "data/siwm/train_annotations.csv"
    test_annotations_file = "data/siwm/test_annotations.csv"
    spoof_type = "replay"

    # Load train and test annotation files into dataframes
    train_df = pd.read_csv(train_annotations_file)
    test_df = pd.read_csv(test_annotations_file)

    # Verify the leave-one-out protocol
    train_spoof_types = set(
        [os.path.dirname(path) for path in train_df["image_path"]]
    )
    test_spoof_types = set(
        [os.path.dirname(path) for path in test_df["image_path"]]
    )

    assert not any(
        spoof_type in path for path in train_spoof_types
    ), f"Train examples contain the specified spoof type: {spoof_type}"
    assert all(
        spoof_type in path for path in test_spoof_types
    ), "Test examples do not have the specified spoof type."

    print("Leave-one-out protocol is correctly implemented.")
