import os
from typing import cast

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from torchgeo.datamodules import MisconfigurationException
from torchgeo.trainers import BYOLTask, MoCoTask, ObjectDetectionTask, SimCLRTask


experiment_dir = "/home/yichia3/scratch/shradha/output/ssl4eo_benchmark_oli_tirs_toa_cdl_unet_resnet18_0.001_ce_0.001_True"

conf = OmegaConf.load("conf/ssl4eo_benchmark_oli_tirs_toa_cdl.yaml")
command_line_conf = OmegaConf.from_cli()

datamodule: LightningDataModule = instantiate(conf.datamodule)

if isinstance(task, ObjectDetectionTask):
    monitor_metric = "val_map"
    mode = "max"
elif isinstance(task, (BYOLTask, MoCoTask, SimCLRTask)):
    monitor_metric = "train_loss"
    mode = "min"
else:
    monitor_metric = "val_loss"
    mode = "min"

trainer: Trainer = instantiate(
    conf.trainer,
    default_root_dir=experiment_dir,
)

trainer.test(ckpt_path="best", datamodule=datamodule)
