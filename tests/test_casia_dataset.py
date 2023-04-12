import pytest
from torch.utils.data import DataLoader
from torchvision import transforms

from src.preprocessing.casia2 import CASIA2
from src.preprocessing.transforms import FaceRegionRCXT, MetaAddLMSquare

from src.models.cnn import CNN


@pytest.fixture
def train_dataset():
    return CASIA2(
        annotations_path="data/casia/images/train/annotations.json",
        root_dir="data/casia/images/train",
    )


@pytest.fixture
def test_dataset():
    return CASIA2(
        annotations_path="data/casia/images/test/annotations.json",
        root_dir="data/casia/images/test",
    )


@pytest.fixture
def train_dataloader(train_dataset):
    rxct = FaceRegionRCXT()
    lmsq = MetaAddLMSquare()
    transform = transforms.Compose([rxct, lmsq])

    train_dataset.transform = transform
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    return train_dataloader


@pytest.fixture
def test_dataloader(test_dataset):
    rxct = FaceRegionRCXT()
    lmsq = MetaAddLMSquare()
    transform = transforms.Compose([rxct, lmsq])

    test_dataset.transform = transform
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    return test_dataloader


def test_model(train_dataloader, test_dataloader):
    model = CNN()
    model.train()

    sample, label = next(iter(train_dataloader))
    image = sample["image"]
    print(type(image))

    output = model(image)
    print(output)
