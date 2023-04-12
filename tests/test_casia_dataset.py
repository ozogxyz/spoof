from spoof.datasets import CASIA
from torchvision import transforms

from spoof.transforms import FaceRegionRCXT, MetaAddLMSquare
from spoof.utils.data import create_annotations


def test_casia():
    casia = CASIA(
        annotations_path="data/casia/test_annotations.json",
        video_root="data/casia/test",
        img_root="data/casia/images/test",
        extract=False,
        transform=transforms.Compose(
            [MetaAddLMSquare(), FaceRegionRCXT(size=(224, 224))]
        ),
    )

    assert casia

    for i in range(10):
        print(casia[i])

    create_annotations(
        metadata_root="data/casia/test/meta/test",
        extracted_frames_root="data/casia/images/test",
        annotations_path="data/casia/test_annotations.json",
    )

    print("Done")
