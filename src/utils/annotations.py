import os
from pathlib import Path


def create_annotations(data_root, train_file):
    """Generate the train file, which has the following format. torch.Dataset class will need the
    absolute path while reading from cv.imread(absolute_path).

    absolute_path0 label0 absolute_path1 label1 ...
    """
    train_file_buf = open(train_file, "w")
    id_list = os.walk(data_root)
    for root, _, files in id_list:
        files.sort()
        for file in files:
            file = str(Path(root) / Path(file))
            label = Path(file).stem[-1]
            if Path(file).suffix == ".jpg":
                train_file_buf.write(file + " " + label + "\n")


def create_meta(data_root, meta_file):
    """Generate the meta file, which has the following format.

    absolute_path0 label0 absolute_path1 label1 ...
    """
    meta_file_buf = open(meta_file, "w")
    id_list = os.walk(data_root)
    for root, _, files in id_list:
        files.sort()
        for file in files:
            file = str(Path(root) / Path(file))
            label = Path(file).stem[-1]
            if Path(file).suffix == ".json":
                meta_file_buf.write(file + " " + label + "\n")


if __name__ == "__main__":
    data_root = "/Users/motorbreath/mipt/thesis/code/spoof/data/casia/train_frames"
    train_file = "/Users/motorbreath/mipt/thesis/code/spoof/data/train.txt"
    meta_file = "/Users/motorbreath/mipt/thesis/code/spoof/data/meta.txt"
    create_annotations(data_root, train_file)
    create_meta(data_root, meta_file)
