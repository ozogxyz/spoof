from argparse import Namespace
from copy import deepcopy
import json
import numpy as np
import os
import warnings

from hydra.utils import instantiate
import pytorch_lightning as pl
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from spoof.utils.collector import MovingAverageCollector
from spoof.utils.metrics import accuracy, eer, fcl, lcf, LABEL_LIVE


# Lightning prints lots of stuff to STDOUT, might be useful to suppress it
warnings.filterwarnings(
    action="ignore", module="pytorch_lightning.utilities.data"
)
warnings.filterwarnings(
    action="ignore", module="pytorch_lightning.trainer.data_loading"
)


class BaseModule(pl.LightningModule):
    """
    A base class with common methods
    """

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.collector = MovingAverageCollector()
        self.subset = "none"
        hparams = Namespace(**hparams)
        torch.autograd.set_detect_anomaly(True)
        self._setup(hparams)

    def _setup(self, hparams):
        # logging frequency setup
        self._log_freq = getattr(hparams, "log_freq", None)
        self._train_vis_freq = getattr(hparams, "train_vis_freq", None)
        self._val_vis_freq = getattr(hparams, "val_vis_freq", None)

    def on_train_epoch_start(self):
        """
        a callback to be called before training epoch run
        currently it sets up training log/visualization frequency
        """
        self.subset = "train"

        if isinstance(self._train_vis_freq, float):
            self.train_vis_freq = int(
                self._train_vis_freq * self.trainer.num_training_batches
            )
        if isinstance(self._log_freq, float):
            self.log_freq = int(
                self._log_freq * self.trainer.num_training_batches
            )

    def on_validation_epoch_start(self) -> None:
        """
        a callback to be called before training epoch run
        currently it sets up validation visualization frequency (if needed?)
        """
        if isinstance(self._val_vis_freq, float):
            self.val_vis_freq = int(
                self._val_vis_freq * self.trainer.num_val_batches[0]
            )
        return super().on_validation_epoch_start()

    def log_tqdm(self, log_str):
        """
        log to stdout, which works fine with TQDM progress bar on most systems
        """
        tqdm.write(f"[{self.__class__.__name__}] {log_str}")

    def forward(self, *args, **kwargs):
        """
        pass data batch through model and collect its outputs / postprocessing

        """
        raise NotImplementedError(f"Method forward() is pure virtual")

    def training_step(self, batch_dict, batch_idx, **kwargs):
        """
        call .forward(batch_dict, **kwargs), then compute loss and save/print logs
        """
        raise NotImplementedError(f"Method training_step() is pure virtual")

    def validation_step(self, *args, **kwargs):
        """
        compute and aggregate validation metrics
        """
        raise NotImplementedError(f"Method validation_step() is pure virtual")

    def on_validation_start(self) -> None:
        """
        clears overall prediction list, might be helpful for visualization purposes
        prediction list shall be used on validation end to aggregate metrics over all batches
        """
        self.score_list = []

    def configure_optimizers(self):
        """
        configure optimizers and schedulers if they are set up in class constructor
        """
        if getattr(self, "lr_scheduler", None) is not None:
            sched_dict = dict(
                {"scheduler": self.lr_scheduler}, **self.lr_scheduler_run_params
            )
            return [self.optimizer], [sched_dict]

        return [self.optimizer]

    def on_save_checkpoint(self, checkpoint) -> None:
        """
        save model and loss configs to checkpoint for reproducibility
        """
        if hasattr(self.hparams, "model"):
            checkpoint["model_config"] = self.hparams.model
        else:
            self.log_tqdm(
                f"Can't save model config, because attribute 'self.hparams.model' not found"
            )
        if hasattr(self.hparams, "loss"):
            checkpoint["loss_config"] = self.hparams.loss
        else:
            self.log_tqdm(
                f"Can't save loss config, because attribute 'self.hparams.loss' not found"
            )

    def log_performance(self, details, batch_size=None):
        """
        some custom logging which I use, but feel free to modify it however you like
        """
        # aggregate loss and metrics over training epoch
        for k, v in details.items():
            val = v.detach().cpu().mean().item()
            if not np.isnan(val):
                self.collector[k].update([val])

        loss_details = {
            nam: meter.value
            for nam, meter in sorted(
                self.collector._dict.items(), key=lambda x: x[0]
            )
        }

        if self.log_freq > 0 and self.global_step % self.log_freq == 0:
            # log to console
            log_str = (
                f"e:{self.current_epoch+1:03d} b: {self.global_step:06d} || "
            )
            log_str += " | ".join(
                [f"{k}: {v:.3f}" for k, v in loss_details.items()]
            )
            self.log_tqdm(log_str)

            # log to tensorboard
            for k, v in details.items():
                self.logger.experiment.add_scalar(
                    f"{k}", v.mean(), self.global_step
                )

        # log with PL logger
        log_dict = {
            f"{k}": v
            for k, v in sorted(loss_details.items(), key=lambda x: x[0])
        }
        self.log_dict(log_dict, batch_size=batch_size)


class SpoofClassificationSystem(BaseModule):
    def _setup(self, hparams):
        super()._setup(hparams)
        # save helper information
        self.train_batch_size = getattr(hparams, "train_batch_size")
        self.eval_batch_size = getattr(hparams, "eval_batch_size", 32)

        # instantiate model class
        self.model = (
            instantiate(hparams.model)
            if getattr(hparams, "model", None) is not None
            else None
        )
        # instantiate loss estimation class
        self.loss_func = (
            None
            if getattr(hparams, "loss", None) is None
            else instantiate(hparams.loss)
        )

        # TODO: let's hardcode Adam optimizer for simplicity and pass only learning rate and weight decay?
        learning_rate = self.hparams.lr
        weight_decay = getattr(self.hparams, "weight_decay", 0.0)
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # load data
        if getattr(self.hparams, "data", None) is not None:
            data_config = deepcopy(self.hparams.data)
            self.ds_train = instantiate(data_config["train"])
            self.ds_train.name = "train"

            val_subsets = [k for k in data_config.keys() if "val" in k]
            self.list_ds_val = []
            for name in val_subsets:
                self.list_ds_val.append(instantiate(data_config[name]))
                self.list_ds_val[-1].name = name

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        performs data loading before training starts
        """
        is_cuda = "cuda" in self.device.type
        num_workers = 0 if os.name == "nt" else 4
        loader_train = DataLoader(
            self.ds_train,
            batch_size=self.train_batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=is_cuda,
        )
        return loader_train

    def val_dataloader(self):
        """
        performs data loading before validation starts
        returns a list of data loaders for multi-dataset validation
        """
        is_cuda = "cuda" in self.device.type
        num_workers = 0 if os.name == "nt" else 4
        loaders_val = [
            DataLoader(
                elem,
                batch_size=self.eval_batch_size,
                pin_memory=is_cuda,
                num_workers=num_workers,
            )
            for elem in self.list_ds_val
        ]
        return loaders_val

    def forward(self, input_dict):
        # TODO: this is an example implementation
        img_tensor = input_dict["image"]
        output = self.model.forward(img_tensor)
        return output

    def training_step(self, batch_dict, batch_idx, **kwargs):
        batch_size = len(batch_dict["filename"])

        # run forward pass for image tensor
        output = self.forward(batch_dict)

        # estimate total loss
        allvars = dict(batch_dict, **output)
        loss, details = self.loss_func(**allvars)

        if loss is None or torch.isnan(loss):
            return None

        self.log(
            "train_loss",
            loss.detach().cpu().mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        # log metrics
        self.add_metrics(details, allvars)
        self.log_performance(details, batch_size)
        if (
            self.train_vis_freq > 0
            and self.global_step % self.train_vis_freq == 1
        ):
            if hasattr(self.model, "add_tensorboard_logs"):
                self.model.add_tensorboard_logs(
                    self.logger.experiment, int(self.global_step), allvars
                )
        return loss

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.subset = self.trainer.val_dataloaders[dataloader_idx].dataset.name
        return super().on_validation_batch_start(
            batch, batch_idx, dataloader_idx
        )

    @torch.no_grad()
    def validation_step(self, batch_dict, batch_idx, dataloader_idx=0):
        output = self.forward(batch_dict)

        # TODO: note that it might be helpful to hide model live-class score prediction inside a special method
        # instead of estimating it somewhere in training loop separately
        preds = self.model.get_liveness_score(output)
        preds = preds.cpu().detach()
        labels = (batch_dict["label"].cpu() == LABEL_LIVE).long()
        filename_list = batch_dict["filename"]
        self.score_list.extend(
            [
                {
                    "name": filename,
                    "score": preds[idx].item(),
                    "label": labels[idx].item(),
                }
                for idx, filename in enumerate(filename_list)
            ]
        )

    def on_validation_end(self) -> None:
        preds = np.array([elem["score"] for elem in self.score_list])
        labels = np.array([elem["label"] for elem in self.score_list])

        subset = self.subset
        th_eval = getattr(self.hparams, "eval_threshold", 0.5)
        metrics = self.calc_metrics(labels, preds, th_eval)
        self.print_scores(metrics, subset)

        metrics["scores"] = self.score_list
        self.dump_scores(metrics, subset)

    @staticmethod
    def calc_metrics(labels, scores, threshold=0.5):
        eer_value, th_eer = eer(
            np.array(labels) == LABEL_LIVE, np.array(scores)
        )
        metric_dict = {
            "m_acc": accuracy(labels, scores, threshold),
            "m_fcl": fcl(labels, scores, threshold),
            "m_lcf": lcf(labels, scores, threshold),
            "m_th_eer": th_eer,
        }
        if eer_value > -1e-3:
            metric_dict["m_eer"] = eer_value
        return metric_dict

    @torch.no_grad()
    def add_metrics(self, details, allvars):
        """
        adds metrics to dict with batch statistics (loss etc.)
        """
        labels = allvars["label"].cpu().numpy().ravel() == LABEL_LIVE
        scores_tensor = self.model.get_liveness_score(allvars)
        scores = scores_tensor.cpu().numpy().ravel()

        metric_dict = self.calc_metrics(labels, scores)
        details.update(
            {k: torch.Tensor([v]).float() for k, v in metric_dict.items()}
        )

    def dump_scores(self, metrics, subset):
        """
        this is an optional step to dump all predicted scores for further outlier analysis
        can be omitted due to possible overcomplication
        """
        latest_ckpt_bn = f"epoch_{self.trainer.current_epoch:03d}"
        score_file = os.path.join(
            self.trainer._default_root_dir,
            "stats",
            subset,
            latest_ckpt_bn + ".json",
        )
        os.makedirs(os.path.dirname(score_file), exist_ok=True)
        with open(score_file, "w") as f:
            json.dump(metrics, f, indent=4)

    def print_scores(self, metrics, subset):
        log_str = f"{subset:40}: "
        log_str += " | ".join(
            [f"{metric}: {score:.3f}" for metric, score in metrics.items()]
        )
        self.log_tqdm(log_str)
        self.log_tqdm("")
