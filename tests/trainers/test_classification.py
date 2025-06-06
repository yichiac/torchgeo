# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import (
    BigEarthNetDataModule,
    EuroSATDataModule,
    MisconfigurationException,
    QuakeSetDataModule,
)
from torchgeo.datasets import BigEarthNet, EuroSAT, QuakeSet, RGBBandsMissingError
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask, MultiLabelClassificationTask


class ClassificationTestModel(Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 10, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes) if num_classes else nn.Identity()
        self.num_features = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PredictBinaryDataModule(QuakeSetDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = QuakeSet(split='test', **self.kwargs)


class PredictMulticlassDataModule(EuroSATDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = EuroSAT(split='test', **self.kwargs)


class PredictMultilabelDataModule(BigEarthNetDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = BigEarthNet(split='test', **self.kwargs)


def create_model(*args: Any, **kwargs: Any) -> Module:
    return ClassificationTestModel(**kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    return None


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class TestClassificationTask:
    @pytest.mark.parametrize(
        'name',
        [
            'bigearthnet_all',
            'bigearthnet_s1',
            'bigearthnet_s2',
            'eurosat',
            'eurosat100',
            'eurosatspatial',
            'fire_risk',
            'quakeset',
            'patternnet',
            'resisc45',
            'so2sat_all',
            'so2sat_s1',
            'so2sat_s2',
            'so2sat_rgb',
            'treesatai',
            'ucmerced',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        if name.startswith('so2sat') or name == 'quakeset':
            pytest.importorskip('h5py', minversion='3.6')

        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(timm, 'create_model', create_model)

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

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_weight_file(self, checkpoint: str) -> None:
        with pytest.warns(UserWarning):
            ClassificationTask(
                model='resnet18', weights=checkpoint, in_channels=13, num_classes=10
            )

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            ClassificationTask(
                model=mocked_weights.meta['model'],
                weights=mocked_weights,
                in_channels=mocked_weights.meta['in_chans'],
                num_classes=10,
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            ClassificationTask(
                model=mocked_weights.meta['model'],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta['in_chans'],
                num_classes=10,
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        ClassificationTask(
            model=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
            num_classes=10,
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        ClassificationTask(
            model=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
            num_classes=10,
        )

    def test_class_weights(self) -> None:
        """Test class_weights parameter functionality."""
        # Test with list of class weights
        class_weights_list = [1.0, 2.0, 0.5]
        task = ClassificationTask(class_weights=class_weights_list, num_classes=3)
        assert task.hparams['class_weights'] == class_weights_list

        # Test with tensor class weights
        class_weights_tensor = torch.tensor([1.0, 2.0, 0.5])
        task = ClassificationTask(class_weights=class_weights_tensor, num_classes=3)
        assert torch.equal(task.hparams['class_weights'], class_weights_tensor)

        # Test with None (default)
        task = ClassificationTask(num_classes=3)
        assert task.hparams['class_weights'] is None

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(EuroSATDataModule, 'plot', plot)
        datamodule = EuroSATDataModule(
            root='tests/data/eurosat', batch_size=1, num_workers=0
        )
        model = ClassificationTask(model='resnet18', in_channels=13, num_classes=10)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(EuroSATDataModule, 'plot', plot_missing_bands)
        datamodule = EuroSATDataModule(
            root='tests/data/eurosat', batch_size=1, num_workers=0
        )
        model = ClassificationTask(model='resnet18', in_channels=13, num_classes=10)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_binary_predict(self, fast_dev_run: bool) -> None:
        pytest.importorskip('h5py', minversion='3.6')
        datamodule = PredictBinaryDataModule(
            root='tests/data/quakeset', batch_size=1, num_workers=0
        )
        model = ClassificationTask(
            model='resnet18', in_channels=4, task='binary', loss='bce'
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_multiclass_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictMulticlassDataModule(
            root='tests/data/eurosat', batch_size=1, num_workers=0
        )
        model = ClassificationTask(
            model='resnet18',
            in_channels=13,
            task='multiclass',
            num_classes=10,
            loss='ce',
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_multilabel_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictMultilabelDataModule(
            root='tests/data/bigearthnet/v1', batch_size=1, num_workers=0
        )
        model = ClassificationTask(
            model='resnet18',
            in_channels=14,
            task='multilabel',
            num_labels=19,
            loss='bce',
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize(
        'model_name', ['resnet18', 'efficientnetv2_s', 'vit_base_patch16_384']
    )
    def test_freeze_backbone(self, model_name: str) -> None:
        model = ClassificationTask(
            model=model_name, num_classes=10, freeze_backbone=True
        )
        assert not all([param.requires_grad for param in model.model.parameters()])
        assert all(
            [param.requires_grad for param in model.model.get_classifier().parameters()]
        )


class TestMultiLabelClassificationTask:
    def test_trainer(self, fast_dev_run: bool) -> None:
        datamodule = BigEarthNetDataModule(
            root='tests/data/bigearthnet/v1', batch_size=1, num_workers=0
        )

        with pytest.deprecated_call():
            model = MultiLabelClassificationTask(
                model='resnet18', in_channels=14, num_classes=19, loss='bce'
            )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
