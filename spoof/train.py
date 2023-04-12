# import hydra
# import torch

# from omegaconf import DictConfig
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose

# from spoof.datasets import CASIA
# from spoof.transforms import FaceRegionRCXT, MetaAddLMSquare


# @hydra.main(version_base=None, config_path="../", config_name="config")
# def main(cfg: DictConfig):
#     # Transforms
#     align = FaceRegionRCXT(size=(224, 224))
#     sq = MetaAddLMSquare()
#     transform = Compose([sq, align])

#     train_ds = CASIA(
#         annotations_path=cfg.data.train_annotations,
#         root_dir=cfg.data.train_images,
#         transform=transform,
#     )

#     test_ds = CASIA(
#         annotations_path=cfg.data.test_annotations,
#         root_dir=cfg.data.test_images,
#         transform=transform,
#     )

#     train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
#     test_dl = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=True)

#     # Load model
#     model = CNN(**cfg.model).to(cfg.device)
#     optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
#     criterion = hydra.utils.instantiate(cfg.criterion)

#     # Train model
#     count = 0
#     print("TRAINING")
#     for epoch in range(cfg.train.epochs):
#         for batch in train_dl:
#             sample, label = batch
#             image = sample["image"].to(cfg.device)
#             label = label.to(cfg.device)

#             # Forward pass
#             print(label)
#             output = model(image)
#             loss = criterion(output, label)

#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()

#             # Update weights
#             optimizer.step()

#             count += 1

#             if count % 10 == 0:
#                 print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
#         break
#     # Save model
#     torch.save(model.state_dict(), "model.pth")


# if __name__ == "__main__":
#     main()
