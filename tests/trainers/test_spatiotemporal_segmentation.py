# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.trainers import SpatioTemporalSegmentationTask


class TestSpatioTemporalSegmentationTask:
    @pytest.mark.parametrize('name', ['pastis_convlstm'])
    def test_trainer(self, name: str, fast_dev_run: bool) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    def test_predict_step(self) -> None:
        task = SpatioTemporalSegmentationTask(
            in_channels=4,
            hidden_dim=8,
            kernel_size=3,
            num_layers=1,
            task='multiclass',
            num_classes=5,
        )
        # (B=2, T=3, C=4, H=16, W=16)
        batch = {'image': torch.randn(2, 3, 4, 16, 16)}
        prediction = task.predict_step(batch, 0)
        assert prediction['probabilities'].shape == (2, 5, 16, 16)

    def test_forward_shape(self) -> None:
        task = SpatioTemporalSegmentationTask(
            in_channels=10,
            hidden_dim=[16, 8],
            kernel_size=[3, (1, 1)],
            num_layers=2,
            task='multiclass',
            num_classes=20,
        )
        x = torch.randn(2, 9, 10, 32, 32)
        y_hat = task(x)
        assert y_hat.shape == (2, 20, 32, 32)
