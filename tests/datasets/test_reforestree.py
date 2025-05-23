# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, ReforesTree


class TestReforesTree:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ReforesTree:
        data_dir = os.path.join('tests', 'data', 'reforestree')
        url = os.path.join(data_dir, 'reforesTree.zip')
        md5 = 'bfc8d35df5d61d1ff4020375843018d9'
        monkeypatch.setattr(ReforesTree, 'url', url)
        monkeypatch.setattr(ReforesTree, 'md5', md5)
        root = tmp_path
        transforms = nn.Identity()
        return ReforesTree(
            root=root, transforms=transforms, download=True, checksum=True
        )

    def test_already_downloaded(self, dataset: ReforesTree) -> None:
        ReforesTree(root=dataset.root, download=True)

    def test_getitem(self, dataset: ReforesTree) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['agb'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3
        assert len(x['bbox_xyxy']) == 2

    def test_len(self, dataset: ReforesTree) -> None:
        assert len(dataset) == 5

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'reforestree', 'reforesTree.zip')
        shutil.copy(url, tmp_path)
        ReforesTree(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'reforesTree.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            ReforesTree(root=tmp_path, checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ReforesTree(tmp_path)

    def test_plot(self, dataset: ReforesTree) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: ReforesTree) -> None:
        x = dataset[0].copy()
        x['prediction_bbox_xyxy'] = x['bbox_xyxy'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()
