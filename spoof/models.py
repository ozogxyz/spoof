import torch
from datasets import CASIA
from torchvision import transforms
from transforms import FaceRegionRCXT, MetaAddLMSquare
from vit_pytorch import ViT
import torch.nn as nn
from torch import optim


def _test_vit():
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )
    ds = CASIA(
        annotations_path="data/casia/test_annotations.json",
        video_root="data/casia/test",
        img_root="data/casia/images/test",
        extract=False,
        transform=transforms.Compose(
            [MetaAddLMSquare(), FaceRegionRCXT(size=(224, 224))]
        ),
    )
    model = model.to("mps")

    # Split the train dataset into train and validation
    train_size = int(0.8 * len(ds))
    valid_size = len(ds) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        ds, [train_size, valid_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = "mps"
    for epoch in range(1):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data["image"]
            data = data.permute(0, 3, 1, 2)  # NCHW

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data["image"]
                data = data.permute(0, 3, 1, 2)  # NCHW
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )


if __name__ == "__main__":
    _test_vit()
