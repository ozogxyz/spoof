import argparse
import copy
import logging

import hydra
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from spoof.dataset.dataset import FaceDataset

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

    # instantiate trainer class, responsible for data loading, compute precision, GPU management, etc.
    trainer = pl.Trainer(
        logger=True,
        callbacks=callbacks,
        replace_sampler_ddp=False,
        benchmark=torch.backends.cudnn.benchmark,
        enable_progress_bar=False,
        accelerator="mps",
        devices=1,
        **params_trainer,
    )

    # Get the list of unique spoof types in the training dataset
    train_dataset = training_system.train_dataloader().dataset
    val_dataset = training_system.val_dataloader().dataset
    # Get the list of unique spoof types in the training dataset, excluding "live"
    spoof_types = train_dataset.annotations[
        train_dataset.annotations["spoof_type"] != "live"
    ]["spoof_type"].unique()

    # Print total counts and lengths of training and validation datasets
    print("Training Dataset:")
    print(train_dataset.annotations["spoof_type"].value_counts())
    print("Total Samples:", len(train_dataset))
    print("-" * 60)  # Separator
    print()

    print("Validation Dataset:")
    print(val_dataset.annotations["spoof_type"].value_counts())
    print("Total Samples:", len(val_dataset))
    print("-" * 60)  # Separator
    print()

    # Iterate over the spoof types
    for spoof_type in spoof_types:
        print("Training for Spoof Type:", spoof_type)
        print("-" * 60)  # Separator

        # Create a copy of the train_dataset annotations
        train_dataset_copy = copy.deepcopy(train_dataset)

        # Create a copy of the val_dataset annotations
        val_dataset_copy = copy.deepcopy(val_dataset)

        # Filter the train_dataset annotations to remove the current spoof type
        train_dataset_copy.leave_out(spoof_type)

        # Filter the val_dataset annotations to add the current spoof type
        val_dataset_copy.leave_out_all_except(spoof_type)

        # Print the length of train_dataset_copy
        print(
            "Train Dataset Length (After Excluding):", len(train_dataset_copy)
        )
        print(train_dataset_copy.annotations["spoof_type"].value_counts())
        print("-" * 60)  # Separator

        # Print the length of val_dataset_copy
        print(
            "Validation Dataset Length (After Including):",
            len(val_dataset_copy),
        )
        print(val_dataset_copy.annotations["spoof_type"].value_counts())
        print("-" * 60)  # Separator
        print()


if __name__ == "__main__":
    args = parse_args()
    train(args)
