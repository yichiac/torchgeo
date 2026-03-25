# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainer for spatio-temporal semantic segmentation."""

from collections.abc import Sequence
from typing import Literal

import torch.nn as nn
from torch import Tensor

from ..models import ConvLSTM
from .base import BaseTask
from .mixins import ClassificationMixin
from .segmentation import SemanticSegmentationTask


class SpatioTemporalSegmentationTask(ClassificationMixin, BaseTask):
    """Spatio-temporal semantic segmentation.

    This task is designed for datasets with image sequences of shape ``(B, T, C, H, W)``.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int | list[int] = 64,
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        num_layers: int = 1,
        task: Literal['binary', 'multiclass', 'multilabel'] = 'multiclass',
        num_classes: int | None = None,
        num_labels: int | None = None,
        labels: list[str] | None = None,
        pos_weight: Tensor | None = None,
        loss: Literal['ce', 'bce', 'jaccard', 'focal', 'dice'] = 'ce',
        class_weights: Tensor | Sequence[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> None:
        """Initialize a new SpatioTemporalSegmentationTask instance.

        Args:
            in_channels: Number of input channels per timestep.
            hidden_dim: Number of hidden channels in each ConvLSTM layer.
            kernel_size: Size of the convolutional kernel in each ConvLSTM layer.
            num_layers: Number of stacked ConvLSTM layers.
            task: One of 'binary', 'multiclass', or 'multilabel'.
            num_classes: Number of prediction classes (only for
                ``task='multiclass'``).
            num_labels: Number of prediction labels (only for
                ``task='multilabel'``).
            labels: List of class names.
            pos_weight: A weight of positive examples used with 'bce' loss.
            loss: Name of the loss function. Supports 'ce', 'bce', 'jaccard',
                'focal', and 'dice'.
            class_weights: Optional rescaling weight given to each class, used
                with 'ce' loss.
            ignore_index: Optional integer class index to ignore in the loss
                and metrics.
            lr: Learning rate for the optimizer.
            patience: Patience for the learning rate scheduler.
        """
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        in_channels: int = self.hparams['in_channels']
        hidden_dim: int | list[int] = self.hparams['hidden_dim']
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = self.hparams[
            'kernel_size'
        ]
        num_layers: int = self.hparams['num_layers']
        num_classes: int = (
            self.hparams['num_classes'] or self.hparams['num_labels'] or 1
        )
        out_channels = hidden_dim if isinstance(hidden_dim, int) else hidden_dim[-1]

        self.model = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        _, last_state_list = self.model(x)
        return self.head(last_state_list[-1][0])

    training_step = SemanticSegmentationTask.training_step
    validation_step = SemanticSegmentationTask.validation_step
    test_step = SemanticSegmentationTask.test_step
    predict_step = SemanticSegmentationTask.predict_step
