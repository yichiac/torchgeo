# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""AgriFieldNet India Challenge dataset."""

from .geo import RasterDataset


class AgriFieldNetMask(RasterDataset):
    """AgriFieldNetMask India Challenge dataset."""
    is_image = False

    def plot