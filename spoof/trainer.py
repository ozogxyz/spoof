import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trainer(cfg, model, train_loader, criterion, optimizer):
    for epoch in range(cfg.train.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.train.epochs}")
        # Train the model
        model.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(cfg.device).transpose(1, 3)
            label = label.to(cfg.device)
            outputs = model(image)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
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
