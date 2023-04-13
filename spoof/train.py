# import copy
# import time

# import hydra
# import torch
# from omegaconf import DictConfig

# from torchvision.transforms import Compose
# from torch import optim
# from datasets import CASIA
# from models import ViT
# from transforms import FaceRegionRCXT, MetaAddLMSquare


# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     device = "mps"
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch}/{num_epochs - 1}")
#         print("-" * 10)

#         model.train()

#         running_loss = 0.0
#         running_corrects = 0

#         for samples, labels in dataloader:
#             img = samples["image"]
#             print(img)
#             break
#             image = image.to(device)
#             labels = labels.to(device)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward
#             outputs = model(image)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # statistics
#             running_loss += loss.item() * image.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)

#         print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

#         # deep copy the model
#         if epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
#     print(f"Best val Acc: {best_acc:4f}")

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model


# @hydra.main(version_base=None, config_path="..", config_name="config")
# def main(cfg: DictConfig) -> None:
#     casia = CASIA(
#         annotations_path=cfg.dataset.annotations_path,
#         video_root=cfg.dataset.video_root,
#         img_root=cfg.dataset.img_root,
#         extract=False,
#         transform=Compose([MetaAddLMSquare(), FaceRegionRCXT(size=(224, 224))]),
#     )
#     train_loader = torch.utils.data.DataLoader(
#         casia,
#         batch_size=cfg.train.batch_size,
#         shuffle=True,
#         num_workers=cfg.train.num_workers,
#     )

#     model = ViT(num_classes=2).to("mps")
#     criterion = torch.nn.BCELoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     model = train_model(model, train_loader, criterion, optimizer, num_epochs=25)


# if __name__ == "__main__":
#     main()
