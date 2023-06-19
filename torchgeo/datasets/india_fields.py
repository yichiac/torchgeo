# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""India Fields 10k dataset."""

import glob
import os
from collections.abc import Sequence
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
from rasterio.crs import CRS
import torch
from torch import Tensor

from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive

class IndiaFields(RasterDataset):
    classes = ["non-fields","fields"]
    filename_glob = "all_bands.tif"
    separate_files = False
    all_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
        "B13"
    ]
    rgb_bands = ["B04", "B03", "B02"]

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Sequence[str] = all_bands,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        self.root = root
        # self.download = download
        # self.checksum = checksum

        bands = bands or self.all_bands
        self.filename_glob = self.filename_glob.format(bands[0])
        # self.filename_regex = self.filename_regex.format(res)

        self._verify()

        super().__init__(
            root, crs=crs, res=res, bands=bands, transforms=transforms, cache=cache
        )

    def _verify(self) -> None:
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            return

        # # Check if the tar.gz files have already been downloaded
        # pathname = os.path.join(self.root, "*.tar.gz")
        # if glob.glob(pathname):
        #     self._extract()
        #     return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        # self._download()
        # self._extract()

    # def _download(self) -> None:
    #     """Download the dataset."""
    #     for biome, md5 in self.md5s.items():
    #         download_url(
    #             self.url.format(biome), self.root, md5=md5 if self.checksum else None
    #         )

    # def _extract(self) -> None:
    #     """Extract the dataset."""
    #     pathname = os.path.join(self.root, "*.tar.gz")
    #     for tarfile in glob.iglob(pathname):
    #         extract_archive(tarfile)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        image = self._merge_files(filepaths, query, self.band_indexes)

        mask_filepaths = []
        for filepath in filepaths:
            mask_filepath = os.path.abspath(os.path.join(filepath, "../..", "mask", "mask.tif"))
            mask_filepaths.append(mask_filepath)

        mask = self._merge_files(mask_filepaths, query)
        # mask_mapping = {0: 0, : 2, 192: 3, 255: 4}

        # for k, v in mask_mapping.items():
        #     mask[mask == k] = v

        sample = {
            "crs": self.crs,
            "bbox": query,
            "image": image.float(),
            "mask": mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)

        # DN = 10000 * REFLECTANCE
        # https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
        image = torch.clamp(image / 10000, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
