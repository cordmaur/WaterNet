import enum
from pathlib import Path
import rasterio as rio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .wnutils import parse_sat_name
from .wnstacimage import WNStacImage
from .wnbaseimage import WNBaseImage, mask_transform
from .wnimagepatcher import WNImagePatcher
from matplotlib.axes import Axes
from patchify import patchify, unpatchify
from typing import Union, Tuple, Optional, Callable, Iterable, List, Sized
import time
from tqdm.notebook import tqdm

from concurrent.futures import ThreadPoolExecutor

import gc


class WNSegmentationItem:

    patch_size = (512, 512)

    def __init__(
        self,
        img: WNStacImage,
        mask: WNBaseImage,
        patch_size: Tuple[int, int] = (512, 512),
        step: int = 262,
    ):
        self.imgpatcher = WNImagePatcher(img=img, patch_size=patch_size, step=step)
        self.maskpatcher = WNImagePatcher(img=mask, patch_size=patch_size, step=step)

        assert len(self.imgpatcher) == len(self.maskpatcher)

        self.status = "Empty"

    @classmethod
    def from_mask(
        cls,
        mask_path: Union[str, Path],
        shape: Optional[Tuple[int, int]] = None,
        patch_size: Tuple[int, int] = (512, 512),
        step: int = 256,
        mask_transform: Optional[Callable] = mask_transform,
    ):
        """
        Given a mask, where the name respects the a satellite naming convention, it will get the corresponding
        Sentinel2 image from MSPC and match both.
        """
        # todo: Include naming convention

        # open the mask (do not load into memory)
        mask = WNBaseImage(path=mask_path, shape=shape, transform=mask_transform)

        # get the mask properties to find correct image
        mask_prop = mask.properties

        if mask_prop is not None:
            dt = "-".join([mask_prop["year"], mask_prop["month"], mask_prop["day"]])
            img = WNStacImage.from_tile(mask_prop["tile"], dt, mask.shape)

            return cls(img, mask, patch_size=patch_size, step=step)

        else:
            raise Exception(f"Properties not found for image {mask_path}")

    @property
    def shape(self):
        return self.maskpatcher.img.shape

    def patchify(self, bands: Iterable[str]):

        # set the status to loading
        self.status = "Loading"

        # create the patches for the images and masks
        self.imgpatcher.patchify(bands)
        self.maskpatcher.patchify()

        self.status = "Loaded"

    def plot(
        self, bands: Iterable[str] = ["B04", "B03", "B02"], figsize: tuple = (20, 10)
    ):

        _, ax = plt.subplots(1, 2, figsize=figsize)

        self.imgpatcher.img.plot(ax=ax[0], bands=bands)
        self.maskpatcher.img.plot(ax=ax[1])

    def plot_segitem_patch(
        self, idx: int, axs=None, figsize: Tuple[int, int] = (10, 10)
    ):
        # Create axes if they are not provided
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=figsize)

        self.imgpatcher.plot_patch(idx=idx, ax=axs[0])
        self.maskpatcher.plot_patch(idx=idx, ax=axs[1])

    def plot_segitem_patches(self, idxs: Union[List[int], np.ndarray], height: int = 3):
        """ """

        n = len(idxs)

        _, axs = plt.subplots(n, 2, figsize=(2.2 * height, n * height))

        for i, idx in enumerate(idxs):
            self.plot_segitem_patch(idx=idx, axs=axs[i, :])  # type: ignore

    def plot_random_patches(self, n: int, height: int = 3):
        """
        Plot n random patches and the corresponding masks
        """
        self.plot_segitem_patches(idxs=np.random.randint(low=0, high=len(self), size=n))

    def decode_batch(self, b):
        return b

    def __len__(self):
        return len(self.imgpatcher)

    def __getitem__(self, idx):

        if self.status == "Empty":
            raise Exception("No patches created. Use .patchify method first.")

        if self.status == "Loading":
            while self.status == "Loading":
                time.sleep(0.1)

        # Before passing the image patch, we will invert the channel position to comply with PyTorch
        return (self.imgpatcher[idx], self.maskpatcher[idx].squeeze())

    def __repr__(self):
        return f"{self.imgpatcher.img.name}->{self.maskpatcher.img.name}"

    def clear(self):
        self.status = "Empty"
        self.imgpatcher.clear()
        self.maskpatcher.clear()

        gc.collect()

    # Criar um status para o SegItem
    # loading -> para evitar lançar 2 processos de loading ao mesmo tempo
    # loaded -> para limpar a memória, caso necessário
    #
