# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""CropHarvest datamodule."""

from torchgeo.datasets import CropHarvest
from .geo import GeoDataModule


class CropHarvestDataModule(GeoDataModule):
    """LightningDataModule implementation for CropHarvest dataset.

    .. versionadded:: 0.9
    """

    def __init__(self, **kwargs):
        """Initialize a new CropHarvestDataModule instance.

        Args:
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CropHarvest`.
        """
        super().__init__(CropHarvest, **kwargs)

    def setup(self, stage: str):
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = CropHarvest(root=self.data_dir, split='train')
        if stage in ['test']:
            self.test_dataset = CropHarvest(root=self.data_dir, split='test')
