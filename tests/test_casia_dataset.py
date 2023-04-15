import cv2
import numpy as np
import pytest
from cv2 import imread
import torch
from torchvision import transforms

from spoof.datasets import CASIA
from spoof.transforms import FaceRegionRCXT, MetaAddLMSquare
from spoof.utils.visualize import draw_face_rectangle, draw_landmarks, show_frame
import timm


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


@pytest.mark.skip(reason="slow")
def test_casia_dataset(casia: CASIA):
    sample, label = casia[5520]
    assert sample["image"].shape == (224, 224, 3)
    assert label == 0


@pytest.mark.skip(reason="visualize")
def test_visualize_casia_dataset(casia: CASIA):
    sample, label = casia[1130]
    image = sample["image"]
    face_rect = sample["meta"]["face_rect"]
    face_landmark = sample["meta"]["face_landmark"]
    draw_face_rectangle(image, face_rect)
    draw_landmarks(image, face_landmark)
    cv2.imshow("ASDFASDF", image)
    cv2.waitKey(0)


@pytest.fixture(autouse=True)
def vit():
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)


def test_vit_inference(vit, casia: CASIA):
    sample, label = casia[1130]
    image = sample["image"]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose(2, 0, 1)
    image = image / 255
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image).float()
    vit.eval()
    with torch.no_grad():
        logits = vit(image)
        probas = torch.softmax(logits[0], dim=0)
        # prints: torch.Size([2])
        print(probas.shape)

    top1_prob = torch.topk(probas, 1)[0].item()
    assert probas.shape == (2,)
    print(top1_prob)
    assert 1 == 2
    assert top1_prob > 0.5
