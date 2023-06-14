import os

import torch
from rasterio.crs import CRS

# from torchgeo.datasets import IndiaFields
# from torchgeo.datamodules import IndiaFieldsDataModule
from torchgeo.datamodules import L7IrishDataModule
from torchgeo.trainers import SemanticSegmentationTask, ObjectDetectionTask
from torchgeo.models import ResNet18_Weights
from torchgeo.samplers import RandomBatchGeoSampler

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader


# root = os.path.join("india_fields10k")
root = os.path.join("l7irish")

weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

# experiment_dir = os.path.join(root, "fields10k_results")
experiment_dir = os.path.join(root, "l7irish_results")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath=experiment_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
csv_logger = CSVLogger(save_dir=experiment_dir, name="pretrained_weights_logs")

task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet18",
    loss="ce",
    weights=weights,
    in_channels=13,
    num_classes=2,
    learning_rate=0.001,
    learning_rate_schedule_patience=6,
    ignore_index=0,
    weight_decay=0,
)
# root="/Users/yc/projects/dali/data/s2_india_fields/imgs",
datamodule = L7IrishDataModule(
    root="/Users/yc/projects/dali/data/l7irish",
    crs=CRS.from_epsg(3857),
    batch_size= 32,
    num_workers=10,
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stopping_callback],
    logger=[csv_logger],
    default_root_dir=experiment_dir,
    min_epochs=1,
    max_epochs=5,
    accelerator="mps",
)

trainer.fit(model=task, datamodule=datamodule)


# sampler = RandomBatchGeoSampler(ds, size=264, batch_size=32)
# dl = DataLoader(ds, batch_sampler=sampler, num_workers=10, )

# test_data = IndiaFields(
    # root="/Users/yc/projects/dali/data/s2_india_fields/imgs",
# )

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# datamodule: LightningDataModule = instantiate(conf.datamodule)
# task: LightningModule = instantiate(conf.module)

# trainer.fit(model=task, datamodule=datamodule)