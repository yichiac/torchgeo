# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import PASTIS
from .geo import NonGeoDataModule


def collate_fn(
    batch: list[dict[str, Tensor]], max_timestamp: int = 61
) -> dict[str, Any]:
    """Custom timeseries collate fn to handle variable length sequences.

    Args:
        batch: list of sample dicts return by dataset
        max_timestamp: maximum length of the time series

    Returns:
        batch dict output

    .. versionadded:: 0.8
    """
    output: dict[str, Any] = {}
    images = [sample['image'] for sample in batch]

    padded_images = []
    for img in images:
        if img.shape[0] < max_timestamp:
            # padding_shape = (max_timestamp - img.shape[0],) + img.shape[1:]
            padding_shape = (max_timestamp - img.shape[0], *img.shape[1:])
            padding = torch.zeros(padding_shape, dtype=img.dtype)
            padded_img = torch.cat([img, padding], dim=0)
            padded_images.append(padded_img)
        else:
            padded_images.append(img)

    output['image'] = torch.stack(padded_images)
    output['mask'] = torch.stack([sample['mask'] for sample in batch])

    if 'bbox_xyxy' in batch[0]:
        output['bbox_xyxy'] = torch.stack([sample['bbox_xyxy'] for sample in batch])
    if 'label' in batch[0]:
        output['label'] = torch.stack([sample['label'] for sample in batch])

    return output


class PASTISDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS dataset.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        max_timestamp: int = 61,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTISDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            max_timestamp: Maximum length of the time series.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS`.
        """
        super().__init__(
            PASTIS, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.max_timestamp = max_timestamp
        self.collate_fn = lambda batch: collate_fn(
            batch, max_timestamp=self.max_timestamp
        )
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = PASTIS(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                1 - self.val_split_pct - self.test_split_pct,
                self.val_split_pct,
                self.test_split_pct,
            ],
            generator,
        )
