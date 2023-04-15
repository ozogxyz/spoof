import os

import cv2
import numpy as np
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from torchvision import transforms
from transforms import FaceRegionRCXT, MetaAddLMSquare

IMG_ROOT = "/Users/motorbreath/mipt/thesis/code/spoof/data/casia/images/train/"
ANN_ROOT = "/Users/motorbreath/mipt/thesis/code/spoof/data/casia/images/train/"


def train_transform(inputs):
    image = inputs.get("image")
    face_rect = inputs.get("face_rect")
    face_landmark = inputs.get("face_landmark")
    label = inputs.get("label")

    # Prepare samples for RCXT class
    meta = {"face_rect": face_rect, "face_landmark": face_landmark}
    sample = {"image": image, "meta": meta}

    # Transform the sample
    align = FaceRegionRCXT(size=(224, 224), crop=(224, 224))
    sq = MetaAddLMSquare()
    transform = transforms.Compose([align, sq])
    sample = transform(sample)

    # Extract the image and label
    image = sample["image"]
    image = image.transpose(1, 2, 0)

    return image, label


def filter_for_data(filename):
    return filename.endswith(".csv")


def sample_reader(inputs):
    """Reads the image and returns a sample dictionary of
    the form:
    {
        "image": image,
        "face_rect": face_rect,
        "face_landmark": face_landmark,
        "label": label,
    }
    """
    filename = inputs.get("image")
    face_rect = inputs.get("face_rect")
    face_landmark = inputs.get("face_landmark")

    label = inputs.get("labels")
    filename = os.path.join(IMG_ROOT, filename)
    image = cv2.imread(filename).transpose(2, 0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sample = {
        "image": image,
        "face_rect": face_rect,
        "face_landmark": face_landmark,
        "label": label,
    }
    return sample


def row_processor(row):
    return {
        "image": row[0],
        "face_rect": np.array(row[1:5]).astype(np.ushort),
        "face_landmark": np.array(row[5:-1]).astype(np.ushort).reshape(-1, 2),
        "labels": int(row[-1]),
    }


def build_datapipes(root_dir="."):
    datapipe = dp.iter.FileLister([root_dir]).filter(filter_for_data)
    datapipe = datapipe.open_files(mode="rt")
    datapipe = datapipe.parse_csv(delimiter=",")
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(row_processor)
    datapipe = datapipe.map(sample_reader)
    datapipe = datapipe.map(train_transform)
    return datapipe


def main():
    datapipe = build_datapipes(ANN_ROOT)
    dl = DataLoader(
        dataset=datapipe,
        batch_size=32,
        shuffle=True,
        num_workers=10,
    )

    for batch in dl:
        print(batch)
        break


if __name__ == "__main__":
    main()
