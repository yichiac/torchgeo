# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""CropHarvest datamodule."""

from typing import Any

from torchgeo.datasets import CropHarvest

from .geo import GeoDataModule


class CropHarvestDataModule(GeoDataModule):
    """LightningDataModule implementation for CropHarvest dataset.

    .. versionadded:: 0.9
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new CropHarvestDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CropHarvest`.
        """
        super().__init__(CropHarvest, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = CropHarvest(root=self.data_dir, split='train')
        if stage in ['test']:
            self.test_dataset = CropHarvest(root=self.data_dir, split='test')
