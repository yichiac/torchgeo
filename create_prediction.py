from torchgeo.models import ResNet18_Weights
from torchgeo.datasets import L7Irish, L8Biome
from torchgeo.datamodules import L7IrishDataModule
from torchgeo.trainers import SemanticSegmentationTask

import torch
from torch.utils.data import DataLoader

PATH = "/scratch.local/yichia3/output/l7irish_output/l7irish_unet_resnet50_0.001_ce_moco/checkpoint-epoch=01-val_loss=0.83.ckpt"
# img_path = "/projects/dali/data/l7irish/subtropical_north/p142_r48/L71142048_04820010422.TIF"
l7_path = "/projects/dali/data/l7irish"
task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet50",
    weights=False,
    in_channels=6,
    num_classes=5,
    loss="ce",
    ignore_index=0,
    learning_rate=1e-4,
    learning_rate_schedule_patience=10,
    freeze_backbone=True,
    freeze_decoder=False,
)

state_dict = torch.load(PATH)
# task.model.encoder.load_state_dict(state_dict)
task.load_from_checkpoint(checkpoint_path=PATH)
# task.predict_step()
# model = L7IrishDataModule.load_from_checkpoint(PATH)

dataset = L7Irish(root=l7_path, download=False, checksum=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

for batch in dataloader:
    x = task.predict(batch)
    print(x)

