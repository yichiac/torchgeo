# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SouthAfricaCropType datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from kornia.contrib import Lambda

from ..datasets import SouthAfricaCropType, random_bbox_assignment
from ..samplers import GridGeoSampler, RandomGeoSampler
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class SouthAfricaCropTypeSatlasDataModule(GeoDataModule):
    """LightningDataModule implementation for the SouthAfricaCropType dataset.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 16,
        length: int | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SouthAfricaCropTypeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SouthAfricaCropType`.
        """
        super().__init__(
            SouthAfricaCropType,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        satlas_std = torch.tensor(
            [3558.0, 3558.0, 3558.0, 8160.0, 8160.0, 8160.0, 8160.0, 8160.0, 8160.0]
        )
        satlas_mean = torch.zeros_like(satlas_std)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=satlas_mean, std=satlas_std),
            K.ImageSequential(Lambda(lambda x: torch.clamp(x, min=0.0, max=1.0))),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=['image', 'mask'],
            extra_args={
                DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}
            },
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=satlas_mean, std=satlas_std),
            K.ImageSequential(Lambda(lambda x: torch.clamp(x, min=0.0, max=1.0))),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = SouthAfricaCropType(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            random_bbox_assignment(dataset, [0.8, 0.1, 0.1], generator)
        )

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