# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Load a SpatioTemporalSegmentationTask checkpoint and run eval + visualization.

Companion to ``spatiotemporal_segmentation_full.py`` for re-running the test
block after interrupting training.

Usage:
    python spatiotemporal_segmentation_full_test.py                 # auto: newest .ckpt
    python spatiotemporal_segmentation_full_test.py path/to.ckpt    # explicit
"""

import glob
import os
import sys

import lightning as L
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer

from torchgeo.datamodules import PASTISDataModule
from torchgeo.datasets import PASTIS
from torchgeo.trainers import SpatioTemporalSegmentationTask

torch.set_float32_matmul_precision('medium')
L.seed_everything(0, workers=True)

root = os.environ.get(
    'PASTIS_ROOT', '/projects/illinois/eng/cs/arindamb/yichia3/pastis'
)
output_dir = '/u/yichia3/torchgeo/outputs_full'

if len(sys.argv) > 1:
    ckpt_path = sys.argv[1]
else:
    ckpts = glob.glob(os.path.join(output_dir, 'lightning_logs/version_*/checkpoints/*.ckpt'))
    if not ckpts:
        raise SystemExit(f'no .ckpt found under {output_dir}')
    ckpt_path = sorted(
        ckpts, key=lambda p: int(p.split('version_')[1].split(os.sep)[0])
    )[-1]
print(f'loading checkpoint: {ckpt_path}')

datamodule = PASTISDataModule(
    root=root,
    bands=PASTIS.s2_bands,
    batch_size=8,
    num_workers=8,
    padding_length=38,
)

task = SpatioTemporalSegmentationTask.load_from_checkpoint(ckpt_path)

trainer = Trainer(
    accelerator='auto',
    precision='bf16-mixed',
    default_root_dir=output_dir,
    logger=False,
)
trainer.test(model=task, datamodule=datamodule)

datamodule.setup('test')
test_dataset = datamodule.test_dataset.dataset

task.eval()

n_samples = 50
pred_dir = os.path.join(output_dir, 'test_predictions')
os.makedirs(pred_dir, exist_ok=True)
saved = 0
for batch in datamodule.test_dataloader():
    if saved >= n_samples:
        break
    lengths = batch['length']
    batch = datamodule.aug(batch)
    with torch.no_grad():
        logits = task(batch['image'], lengths=lengths)
    predictions = logits.argmax(dim=1)
    for i in range(batch['image'].shape[0]):
        if saved >= n_samples:
            break
        sample = {
            'image': batch['image'][i],
            'mask': batch['mask'][i],
            'prediction': predictions[i],
        }
        fig = test_dataset.plot(sample)
        fig.savefig(
            os.path.join(pred_dir, f'test_prediction_{saved}.png'),
            bbox_inches='tight',
        )
        plt.close(fig)
        saved += 1
