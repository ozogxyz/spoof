import logging

import cv2
import numpy as np
import torchdata.datapipes as dp
from torchvision import transforms
from transforms import FaceRegionRCXT, MetaAddLMSquare

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_for_data(filename):
    return filename.endswith(".csv")


def apply_transform(inputs):
    logger.info("ENTERING TRANSFORM")
    image = inputs.get("image")
    logger.info(f"BEFORE TRANSFORM: {image.shape}")
    face_rect = inputs.get("face_rect")
    face_landmark = inputs.get("face_landmark")
    label = inputs.get("label")

    # Prepare samples for RCXT class
    meta = {"face_rect": face_rect, "face_landmark": face_landmark}
    sample = {"image": image, "meta": meta}

    # Transform the sample, these two wants dicts
    align = FaceRegionRCXT(crop=(224, 224))
    sq = MetaAddLMSquare()
    # TODO normalize by channels
    transform = transforms.Compose([align, sq])
    sample = transform(sample)

    # Apply normalization, needs tensor, so separated
    tens = transforms.ToTensor()
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([tens, norm])
    image = sample["image"]
    image = transform(image)

    # Model expects (224, 224, 3)
    image = image.transpose(0, 2)
    logger.info(f"AFTER TRANSFORM: {image.shape}")

    return image, label


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
    # logger.info("ENTERING SAMPLE_READER")
    filename = inputs.get("image")
    face_rect = inputs.get("face_rect")
    face_landmark = inputs.get("face_landmark")

    label = inputs.get("labels")
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sample = {
        "image": image,
        "face_rect": face_rect,
        "face_landmark": face_landmark,
        "label": label,
    }
    # logger.info("LEAVING SAMPLE_READER")
    return sample


def row_processor(row):
    return {
        "image": row[0],
        "face_rect": np.array(row[1:5]).astype(np.float16),
        "face_landmark": np.array(row[5:-1]).astype(np.float16).reshape(-1, 2),
        "labels": int(row[-1]),
    }


def build_datapipes(root_dir):
    datapipe = dp.iter.FileLister([root_dir]).filter(filter_for_data)
    datapipe = datapipe.open_files(mode="rt")
    datapipe = datapipe.parse_csv(delimiter=",")
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(row_processor)
    datapipe = datapipe.map(sample_reader)
    datapipe = datapipe.map(apply_transform)

    return datapipe
