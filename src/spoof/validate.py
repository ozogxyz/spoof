import argparse
import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import yaml

from spoof.dataset import threaded_loader
from spoof.training_system import SpoofClassificationValidator

logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Validation/testing script")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to model checkpoint"
    )
    parser.add_argument(
        "--config-data", nargs="+", type=str, help="path to data config"
    )
    parser.add_argument(
        "--print-threshold",
        default=1.01,
        type=float,
        help="Performance print threshold for corner cases",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose mode: more logs",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="validation data loader batch size",
    )
    parser.add_argument(
        "--threads",
        default=4,
        type=int,
        help="validation data loader num threads",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="Device ID in case of multiple GPUs",
    )
    return parser.parse_args()


def validate(args):
    logging.getLogger("pytorch_lightning").setLevel(
        logging.INFO if args.verbose else logging.WARNING
    )
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    cuda_str = "cuda"
    if bool(args.device):
        cuda_str = f"cuda:{args.device}"
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    logger.info(f"running on device: {device}")

    # data source parsing
    data_config_dict = {}
    for path_config_data in args.config_data:
        with open(path_config_data, "r") as f:
            data_config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

    # model config loading
    logger.info(f"loading model: {args.ckpt}")
    ckpt_dict = torch.load(
        args.ckpt, map_location=lambda storage, loc: storage
    )
    model_config = ckpt_dict["model_config"]

    output_dir = Path(args.ckpt).parent
    params_runner = {
        "default_root_dir": output_dir,
        "gpus": [args.device] if args.device >= 0 else -1,
        "log_every_n_steps": 10,
        "accelerator": "mps",
    }
    params = {
        "model": model_config,
        **{
            k: args.__dict__[k] for k in ["ckpt", "print_threshold", "verbose"]
        },
        "trainer_params": params_runner,
    }
    extractor = SpoofClassificationValidator(**params)

    runner = pl.Trainer(
        logger=True, replace_sampler_ddp=False, **params_runner
    )
    for subset_name, subset_ds_config in data_config_dict.items():
        subset_ds = hydra.utils.instantiate(subset_ds_config)
        subset_ds.name = subset_name

        data_loader = threaded_loader(
            subset_ds,
            device != torch.device("cpu"),
            batch_size=args.batch_size,
            threads=args.threads,
        )

        extractor.hparams.subset = subset_name
        runner.validate(
            extractor,
            dataloaders=data_loader,
            verbose=False,
            ckpt_path=args.ckpt,
        )


def main():
    args = parse_args()
    validate(args)


if __name__ == "__main__":
    main()
    exit(0)
