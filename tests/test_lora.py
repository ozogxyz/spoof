import hydra
import pytorch_lightning as pl


def test_lora(config_training):
    conf_train_sys = config_training["training_system"]
    conf_train_sys["trainer_params"] = config_training["trainer_params"]
    conf_train_sys["train_batch_size"] = 1
    train_sys = hydra.utils.instantiate(conf_train_sys, _recursive_=False)

    params_trainer = conf_train_sys["trainer_params"]
    params_trainer["max_epochs"] = 1
    params_trainer["default_root_dir"] = "/tmp/test_lora"
    params_trainer["fast_dev_run"] = True
    params_trainer["accelerator"] = "mps"

    trainer = pl.Trainer(**params_trainer)

    trainer.fit(train_sys)
