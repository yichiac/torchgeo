# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Sentinel-2 and CDL datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from matplotlib.figure import Figure

from ..datasets import (
    CDL,
    NCCM,
    AgriFieldNet,
    Sentinel2,
    SouthAfricaCropType,
    SouthAmericaSoybean,
    random_bbox_assignment,
)
from ..samplers import GridGeoSampler, RandomGeoSampler
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class Sentinel2CDLOOD(GeoDataModule):
    """LightningDataModule implementation for the Sentinel-2 and CDL datasets.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 64,
        length: int | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sentinel2CDLDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CDL` (prefix keys with ``cdl_``) and
                :class:`~torchgeo.datasets.Sentinel2`
                (prefix keys with ``sentinel2_``).
        """
        # Define prefix for Cropland Data Layer (CDL) and Sentinel-2 arguments
        cdl_signature = 'cdl_'
        sentinel2_signature = 'sentinel2_'
        self.cdl_kwargs = {}
        self.sentinel2_kwargs = {}

        for key, val in kwargs.items():
            if key.startswith(cdl_signature):
                self.cdl_kwargs[key[len(cdl_signature) :]] = val
            elif key.startswith(sentinel2_signature):
                self.sentinel2_kwargs[key[len(sentinel2_signature) :]] = val

        super().__init__(
            CDL, batch_size, patch_size, length, num_workers, **self.cdl_kwargs
        )

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=torch.tensor(10000)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=['image', 'mask'],
            extra_args={
                DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}
            },
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=torch.tensor(10000)), data_keys=['image', 'mask']
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.sentinel2 = Sentinel2(**self.sentinel2_kwargs)
        self.cdl = CDL(**self.cdl_kwargs)
        self.nccm = NCCM()
        self.agrifieldnet = AgriFieldNet()
        self.south_africa_crop_type = SouthAfricaCropType()
        self.south_america_soybean = SouthAmericaSoybean()

        self.train_val_dataset = self.sentinel2 & (self.nccm|self.agrifieldnet|self.south_africa_crop_type|self.south_america_soybean)

        generator = torch.Generator().manual_seed(0)

        (self.train_dataset, self.val_dataset) = (
            random_bbox_assignment(
                self.train_val_dataset, [0.8, 0.2], generator=generator
            )
        )
        self.test_dataset = self.sentinel2 & self.cdl

        if stage in ['fit']:
            self.train_sampler = RandomGeoSampler(
                self.train_dataset, self.patch_size, self.length
            )
        if stage in ['fit', 'validate']:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ['test']:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        """Run CDL plot method.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.
        """
        return self.cdl.plot(*args, **kwargs)