import os


from src.dataset.extract_frames import create_labels_csv, extract_frames


def test_extract_frames_for_train(train_src, train_metadata_src, train_dest):
    extract_frames(train_src, train_metadata_src, dest=train_dest)

    assert os.path.exists(train_dest)
    assert os.path.exists(os.path.join(train_dest, "labels.csv"))

    with open(os.path.join(train_dest, "labels.csv")) as f:
        lines = f.readlines()
        assert len(lines) == 1000


def test_extract_frames_for_test(test_src, test_metadata_src, test_dest):
    extract_frames(test_src, test_metadata_src, dest=test_dest)

    assert os.path.exists(test_dest)
    assert os.path.exists(os.path.join(test_dest, "labels.csv"))

    with open(os.path.join(test_dest, "labels.csv")) as f:
        lines = f.readlines()
        assert len(lines) == 1000
