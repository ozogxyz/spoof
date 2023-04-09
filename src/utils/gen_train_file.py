import os


def gen_train_file(data_root, train_file):
    """Generate the train file, which has the following format.

    relative_path0 label0 relative_path1 label1 relative_path2 label2 ...
    """
    train_file_buf = open(train_file, "w")
    id_list = os.listdir(data_root)
    id_list.sort()
    for label, id_name in enumerate(id_list):
        cur_id_folder = os.path.join(data_root, id_name)
        cur_img_list = os.listdir(cur_id_folder)
        cur_img_list.sort()
        for index, image_name in enumerate(cur_img_list):
            image_path = os.path.join(id_name, image_name)
            train_file_buf.write(image_path + " " + str(label) + "\n")


if __name__ == "__main__":
    data_root = "/Users/motorbreath/mipt/thesis/datasets/casia/train/data/train"
    train_file = "/Users/motorbreath/mipt/thesis/code/spoof/data/train.txt"
    gen_train_file(data_root, train_file)
