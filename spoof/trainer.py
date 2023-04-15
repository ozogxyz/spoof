import logging

import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trainer(cfg, model, train_loader, criterion, optimizer):
    for epoch in range(cfg.train.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.train.epochs}")
        # Train the model
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(cfg.device).transpose(1, 3)
            labels = labels.to(cfg.device).float()

            # Forward pass
            outputs = model(images)

            # Get predictions
            preds = torch.argmax(outputs, dim=1).float()

            # Calculate loss
            loss = criterion(preds, labels)
            optimizer.zero_grad()

            # Backward pass
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if (i + 1) % cfg.train.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{cfg.train.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
            logger.info(
                "Epoch [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, cfg.train.epochs, loss.item()
                )
            )
        logger.info("Training phase complete for epoch {}".format(epoch + 1))
