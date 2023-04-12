import cv2
import pytest
from cv2 import imread
from torchvision import transforms

from spoof.datasets import CASIA
from spoof.transforms import FaceRegionRCXT, MetaAddLMSquare
from spoof.utils.visualize import draw_face_rectangle, draw_landmarks, show_frame


@pytest.fixture(scope="module", autouse=True)
def casia():
    return CASIA(
        annotations_path="data/casia/test_annotations.json",
        video_root="data/casia/test",
        img_root="data/casia/images/test",
        extract=False,
        transform=transforms.Compose(
            [MetaAddLMSquare(), FaceRegionRCXT(size=(224, 224))]
        ),
    )


def test_casia_dataset(casia: CASIA):
    sample, label = casia[5520]
    assert sample["image"].shape == (224, 224, 3)
    assert label == 0


def test_visualize_casia_dataset(casia: CASIA):
    sample, label = casia[1130]
    image = sample["image"]
    face_rect = sample["meta"]["face_rect"]
    face_landmark = sample["meta"]["face_landmark"]
    draw_face_rectangle(image, face_rect)
    draw_landmarks(image, face_landmark)
    cv2.imshow("ASDFASDF", image)
    cv2.waitKey(0)
