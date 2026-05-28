# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Spatiotemporal Segmentation with ConvLSTM on PASTIS100.

Script form of docs/tutorials/spatiotemporal_segmentation.ipynb.
"""

import os

import lightning as L
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from torchgeo.datamodules import PASTIS100DataModule
from torchgeo.datasets import PASTIS100
from torchgeo.trainers import SpatioTemporalSegmentationTask

torch.set_float32_matmul_precision('medium')
L.seed_everything(0, workers=True)

root = os.environ.get(
    'PASTIS100_ROOT', '/projects/illinois/eng/cs/arindamb/yichia3/pastis100'
)
# output_dir = os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.abspath(__file__)))
output_dir = '/u/yichia3/torchgeo/outputs'
os.makedirs(root, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

dataset = PASTIS100(root=root, bands=PASTIS100.s2_bands, download=False)
fig = dataset.plot(dataset[91], suptitle='PASTIS100 sample')
fig.savefig(os.path.join(output_dir, 'pastis100_sample.png'), bbox_inches='tight')
plt.close(fig)
print(f'{len(dataset)} patches, {tuple(dataset[0]["image"].shape)} (T, C, H, W)')

# Sqrt-inverse-frequency class weights, normalized to mean 1. Milder than
# pure inverse frequency (which causes mode collapse) but still corrects
# the strong background dominance.
num_classes = 20
ignore_index = 19
class_counts = torch.zeros(num_classes, dtype=torch.long)
for i in range(len(dataset)):
    mask = dataset[i]['mask'].flatten().long()
    class_counts += torch.bincount(mask, minlength=num_classes)
class_weights = torch.zeros(num_classes)
valid_mask = torch.arange(num_classes) != ignore_index
valid_counts = class_counts[valid_mask].float().clamp(min=1)
w = 1.0 / valid_counts.sqrt()
w = w / w.mean()
class_weights[valid_mask] = w
print('class weights:', class_weights.tolist())

datamodule = PASTIS100DataModule(
    root=root, bands=PASTIS100.s2_bands, batch_size=4, num_workers=4, padding_length=43
)

task = SpatioTemporalSegmentationTask(
    model='convlstm',
    in_channels=10,
    num_classes=num_classes,
    ignore_index=ignore_index,
    loss='ce',
    class_weights=class_weights,
    lr=1e-4,
    hidden_dim=128,
    num_layers=2,
    kernel_size=3,
    head_kernel_size=3,
)

trainer = Trainer(
    max_epochs=100,
    accelerator='auto',
    log_every_n_steps=1,
    callbacks=[
        EarlyStopping('val_loss', patience=20),
        ModelCheckpoint(monitor='val_loss', mode='min'),
    ],
    default_root_dir=output_dir,
)
trainer.fit(task, datamodule=datamodule)

trainer.test(model=task, datamodule=datamodule, ckpt_path='best')

datamodule.setup('test')
test_dataset = datamodule.test_dataset.dataset

task.eval()

batch = next(iter(datamodule.test_dataloader()))
lengths = batch['length']
batch = datamodule.aug(batch)

with torch.no_grad():
    logits = task(batch['image'], lengths=lengths)
predictions = logits.argmax(dim=1)

for i in range(3):
    sample = {
        'image': batch['image'][i],
        'mask': batch['mask'][i],
        'prediction': predictions[i],
    }
    fig = test_dataset.plot(sample, suptitle=f'Test sample {i}')
    fig.savefig(
        os.path.join(output_dir, f'test_prediction_{i}.png'), bbox_inches='tight'
    )
    plt.close(fig)
