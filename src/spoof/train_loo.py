import argparse
import copy
import json
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example of training script for pytorch lightning"
    )
    parser.add_argument(
        "--cfg-training",
        default="config/train.yaml",
        type=str,
        help="Training config file",
    )
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="train batch size"
    )
    parser.add_argument(
        "-e", "--epochs", default=1, type=int, help="train epoch count"
    )
    parser.add_argument(
        "-t",
        "--train_dir",
        default="logs",
        type=str,
        help="path to dir to save experiment result",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="path to pretrained checkpoint to continue training from",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="number of GPU to use",
    )
    return parser.parse_args()


def train(args: argparse.Namespace):
    # useful trick for some GPUs
    torch.set_float32_matmul_precision("high")  # | 'high')

    # disable lots of pytorch lightning logs, set to logging.INFO for verbosity if needed
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    output_dir = "logs/stats/"
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "metrics.log"))
    logger.addHandler(file_handler)

    # read training config with keys 'model', 'loss' and 'data'
    path_config_training = args.cfg_training
    with open(path_config_training, "r") as f:
        config_training = yaml.load(f, Loader=yaml.FullLoader)

    config_training_system = config_training["training_system"]
    # set hyper parameters
    config_training_system["trainer_params"] = config_training[
        "trainer_params"
    ]
    config_training_system["train_batch_size"] = args.batch_size

    # instantiate PL training system, containing loss function, model, data and training/validation loops
    training_system = hydra.utils.instantiate(
        config_training_system, _recursive_=False
    )

    # prepare params for trainer class
    params_trainer = config_training_system["trainer_params"]
    params_trainer["max_epochs"] = args.epochs
    params_trainer["default_root_dir"] = args.train_dir
    if args.device is not None:
        params_trainer["devices"] = [args.device]
        params_trainer["accelerator"] = "gpu"

    # # training callbacks, e.g. model checkpoint saving or TQDM progress bar
    logger.info(
        f"default checkpoint dir: {params_trainer['default_root_dir']}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=params_trainer["default_root_dir"],
        filename="ep{epoch:03d}_loss{train_loss:.2f}_acc{m_acc:.3f}_eer{m_eer:.3f}",
        save_top_k=-1,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_callback]

    # Get the list of unique spoof types in the training dataset
    train_dataset = training_system.train_dataloader().dataset
    val_dataset = training_system.val_dataloader().dataset

    # Get the list of unique spoof types in the training dataset, excluding "live"
    spoof_types = train_dataset.annotations[
        train_dataset.annotations["spoof_type"] != "live"
    ]["spoof_type"].unique()

    aggregated_metrics = {}
    # Iterate over the spoof types
    for spoof_type in spoof_types:
        trainer = pl.Trainer(
            logger=True,
            callbacks=callbacks,
            replace_sampler_ddp=False,
            benchmark=torch.backends.cudnn.benchmark,
            enable_progress_bar=False,
            **params_trainer,
        )

        logger.info("Training for Spoof Type: %s", spoof_type)
        logger.info("-" * 60)  # Separator

        # Copy and update the datasets
        train_dataset_copy = copy.deepcopy(train_dataset)
        train_dataset_copy.leave_out(spoof_type)
        training_system.train_dataset = train_dataset_copy

        val_dataset_copy = copy.deepcopy(val_dataset)
        val_dataset_copy.leave_out_all_except(spoof_type)
        training_system.val_dataset = val_dataset_copy
        training_system.ds_val.name = spoof_type

        # Train the model
        trainer.fit(training_system)

        # Calculate average metrics
        train_metrics = trainer.callback_metrics
        logger.info(train_metrics)
        print(train_metrics)

        # Add the average metrics to the aggregated_metrics dictionary with the spoof type as the key
        aggregated_metrics[spoof_type] = train_metrics

    # Log the aggregated_metrics dictionary
    logger.info("-" * 60)  # Separator
    logger.info("Aggregated Metrics")
    logger.info(aggregated_metrics)

    # Average the aggregated_metrics dictionary
    average_metrics = {
        spoof_type: {
            metric: torch.mean(values).item()
            for metric, values in metrics.items()
        }
        for spoof_type, metrics in aggregated_metrics.items()
    }

    logger.info("-" * 60)  # Separator
    logger.info("Average Metrics")
    logger.info(average_metrics)

    with open("logs/stats/average_metrics.json", "w") as f:
        f.write(json.dumps(average_metrics, indent=4))

    json_metrics = {}

    for spoof_type, metrics in aggregated_metrics.items():
        json_metrics[spoof_type] = {}
        for metric_name, metric_value in metrics.items():
            json_metrics[spoof_type][metric_name] = metric_value.item()

    with open("logs/stats/aggregated_metrics.json", "w") as f:
        f.write(json.dumps(json_metrics, indent=4))


if __name__ == "__main__":
    args = parse_args()
    train(args)
