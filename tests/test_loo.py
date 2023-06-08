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

    for spoof_type in spoof_types:
        train_dataset = FaceDataset(annotations_file=train_annotations_file)
        train_dataset.leave_out(spoof_type)
        train_new_length = len(train_dataset)

        test_dataset = FaceDataset(annotations_file=test_annotations_file)
        test_dataset.leave_out_all_except(spoof_type)
        test_new_length = len(test_dataset)

        assert train_new_length == len(
            train_annotations[train_annotations["spoof_type"] != spoof_type]
        )
        assert test_new_length == len(
            test_annotations[test_annotations["spoof_type"] == spoof_type]
            + test_annotations[test_annotations["spoof_type"] == "live"]
        )
