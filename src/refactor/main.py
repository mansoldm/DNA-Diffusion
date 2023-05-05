import logging
import os
import sys
from dataclasses import dataclass

import hydra
# import pyrootutils
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class DNADiffusionConfig:
    data: str = "vanilla_sequences"
    model: str = "dnadiffusion"
    logger: str = "wandb"
    trainer: str = "default"
    callbacks: str = "default"
    paths: str = "default"
    seed: int = 42
    train: bool = True
    test: bool = False
    # ckpt_path: None


cs = ConfigStore.instance()
cs.store(name="dnadiffusion_config", node=DNADiffusionConfig)


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DNADiffusionConfig):
    # print(HydraConfig.get().job.name)

    # run = wandb.init(
    #    name=parser.logdir,
    #    save_dir=parser.logdir,
    #    project=cfg.logger.wandb.project,
    #    config=cfg,
    # )

    # Placeholder for what loss or metric values we plan to track with wandb
    # wandb.log({"loss": cfg.model.criterion})
    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {get_original_cwd()}")

    pl.seed_everything(cfg.seed)
    # Check if this works
    # model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    unet = instantiate(cfg.model.unet)
    optimizer = instantiate(cfg.model.optimizer, params=unet.parameters())
    lr_scheduler = instantiate(cfg.model.lr_scheduler, optimizer=optimizer)

    model = instantiate(
        cfg.model,
        unet=unet, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    if cfg.ckpt_path:
        model.load_from_checkpoint(cfg.ckpt_path)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=10,
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")


    # logger = instantiate(cfg.logger)
    trainer = pl.Trainer(
        callbacks=[model_checkpoint_callback, lr_monitor_callback],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices
    )

    trainer.fit(
        model, 
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader()
    )


if __name__ == "__main__":
    main()
