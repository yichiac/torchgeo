import os

from torchgeo.trainers import SemanticSegmentationTask
import satlaspretrain_models
# from satlaspretrain_models.utils import SatlasPretrain_weights
import torch

from lightning.pytorch import Trainer
from torchgeo.datamodules import Sentinel2CDLDataModule

batch_size = 128
patch_size = 256
num_workers = 0
max_epochs = 10
fast_dev_run = False

datamodule = Sentinel2CDLDataModule(
    batch_size = batch_size, patch_size = patch_size, num_workers = num_workers,
    cdl_paths = "/data/yichiac/cdl_harmonized_block",
    cdl_years = [2023],
    sentinel2_paths = "/data/yichiac/cdl_2023_small",
    sentinel2_cache =False,
    sentinel2_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
)

print('finish loading datamodule')

weights_manager = satlaspretrain_models.Weights()

class SatlasSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, model_identifier: str, fpn: bool = True, pretrained: bool = True, *args, **kwargs):
        self.model_identifier = model_identifier
        self.pretrained = pretrained
        self.fpn = fpn
        super().__init__(*args, **kwargs)

    def configure_models(self):
        self.model = weights_manager.get_pretrained_model(
            model_identifier=self.model_identifier,
            fpn=self.fpn,
            head=satlaspretrain_models.Head.SEGMENT,
            num_categories=self.hparams["num_classes"]
        )
        # first_layer = self.model.features[0][0]
        head_in_channels = self.model.head.layers[0][0].in_channels
        self.model.head.layers[0][0].in_channels = torch.nn.Conv2d(9,
                                    head_in_channels.out_channels,
                                    kernel_size=head_in_channels.kernel_size,
                                    stride=head_in_channels.stride,
                                    padding=head_in_channels.padding,
                                    bias=(head_in_channels.bias is not None))
        self.model.head = torch.nn.Linear(in_features=1024, out_features=self.hparams["num_classes"], bias=True)

        # head_in_channels = self.model.head.layers[0][0].in_channels
        # upconv = torch.nn.Sequential(
        #     torch.nn.Conv2d(head_in_channels, head_in_channels, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
        #     torch.nn.Conv2d(head_in_channels, head_in_channels, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
        # )

        # place upconv layers before provided head layers
        # modules = list(upconv.children()) + list(self.model.head.layers.children())
        # self.model.head.layers = torch.nn.Sequential(*modules)

        # Freeze backbone and unfreeze classifier head
        # if self.hparams["freeze_backbone"]:
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        #     for param in self.model.head.parameters():
        #         param.requires_grad = True

    def on_validation_epoch_end(self):
        # Accessing metrics logged during the current validation epoch
        val_loss = self.trainer.callback_metrics.get('val_loss', 'N/A')
        val_acc = self.trainer.callback_metrics.get('val_OverallAccuracy', 'N/A')
        print(f"Epoch {self.current_epoch} Validation - Loss: {val_loss}, Accuracy: {val_acc}")

task = SatlasSemanticSegmentationTask(
    num_classes=6,
    freeze_backbone=True,
    model_identifier="Sentinel2_Resnet50_SI_MS",
    pretrained=True,
    fpn=False,
    ignore_index=0,
)


accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = Trainer(
    accelerator=accelerator,
    default_root_dir="/home/yichiac/torchgeo",
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    min_epochs=1,
    max_epochs=max_epochs,
)

trainer.fit(model=task, datamodule=datamodule)