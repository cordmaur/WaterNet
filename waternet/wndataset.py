from pathlib import Path
from unittest.mock import patch
import rasterio as rio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .wnutils import parse_sat_name
from .wnstacimage import WNStacImage
from .wnbaseimage import WNBaseImage
from .wnsegmantationitem import WNSegmentationItem

from patchify import patchify, unpatchify
from typing import Union, Optional, Tuple, List
import time

from tqdm.notebook import tqdm

from concurrent.futures import ThreadPoolExecutor

import gc


class WNDataSet:
    def __init__(self, items: List[WNSegmentationItem]):
        """
        This function should not be called directly. Call .from_masks instead.
        """
        self.items = items
        self._bands: Optional[list] = None

        # create a pool for executing concurrent tasks
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.processes = []

    @classmethod
    def from_masks(
        cls,
        masks_path: Union[str, Path],
        patch_size: tuple = (512, 512),
        step: int = 262,
        pattern: str = "*",
        shape: Optional[Tuple[int, int]] = None,
        retries: int = 3,
    ):
        """
        Create a new WNDataset instance from the masks. It match the masks with image IDs for quick access to the MSPC.
        :out_shape: if None, will load everything in the resolution of the mask, otherwise, will force a shape
        :patch_size: size for the patches that will be created for training the model
        :retries: number of retries if pc api is not responding to perform the matching
        """

        # get the masks
        path = Path(masks_path)
        assert path.exists()

        # create SegmentationItems for each mask
        masks = list(path.rglob(pattern))
        items = []

        for mask in tqdm(
            masks,
            desc="Matching imgs/masks",
        ):
            retry = retries

            while retry > 0:
                try:
                    item = WNSegmentationItem.from_mask(
                        mask_path=mask,
                        shape=shape,
                        patch_size=patch_size,
                        step=step,
                    )
                    items.append(item)
                    retry = 0

                except Exception as e:
                    print(e)
                    retry = retry - 1

        return cls(items)

    def __len__(self):
        return sum(map(len, self.items))

    @property
    def bands(self):
        # Cannot proceed if bands is not specified
        if self._bands is None:
            raise Exception("Bands not set. Set desired bands first.")
        else:
            return self._bands

    @bands.setter
    def bands(self, value: list):

        self._bands = value

    @staticmethod
    def finished_loading(future):
        print("Loaded sucessfully")

    def _previous_item(self, idx):
        return idx - 1 if idx > 0 else len(self.items) - 1

    def _next_item(self, idx):
        return idx + 1 if idx < len(self.items) - 1 else 0

    def _load_next(self, idx):

        next_idx = self._next_item(idx)

        # if there is just 1 item, do nothing
        if len(self.items) == 1:
            return

        # if the next position is empty, load it in parallel
        if self.items[next_idx].status == "Empty":
            self.items[next_idx].status = "Loading"
            print(f"Loading image {next_idx} in background")
            process = self.pool.submit(self.items[next_idx].patchify, self.bands)
            process.add_done_callback(WNDataSet.finished_loading)
            self.processes.append(process)

    def get_item_patch_pos(self, idx):
        """
        This function returns, for a given idx, the corresponding segitem and patch index within the segitem
        """
        if idx >= len(self):
            raise Exception("Index out of bounds of the dataset")

        # in terms of int_position (which item) and rel_position (which patch within the item)
        breaks = np.cumsum(list(map(len, self.items)))
        segitem_idx = (breaks <= idx).sum()
        breaks = np.insert(breaks, 0, 0)
        patch_idx = idx - breaks[segitem_idx]

        return segitem_idx, patch_idx

    def _clear_previous(self, idx):

        # Get the indices for the previous and next items
        previous_idx = self._previous_item(idx)

        # if the list has just 1 or 2 elements, we can skip this function
        # as there will be no loading/clearing along the way
        if len(self.items) > 2:
            if self.items[previous_idx].status == "Loaded":
                print(f"Cleared image {previous_idx}")
                self.items[previous_idx].clear()

    def __getitem__(self, idx):

        # first, find the item for this specific idx
        segitem_idx, patch_idx = self.get_item_patch_pos(idx)

        # with the int_position, patchify the corresponding segmentation_item (if needed)
        item = self.items[segitem_idx]
        if item.status == "Empty":
            item.patchify(self.bands)

        # then check for the next and previous images
        self._clear_previous(segitem_idx)
        self._load_next(segitem_idx)

        # wait if it is still loading
        while item.status == "Loading":
            time.sleep(0.01)

        return item[patch_idx]

    def clear(self, slice: slice = slice(None)):
        for item in self.items[slice]:
            item.clear()

        self.processes = []

    def decode_batch(self, b):
        return b

    @property
    def loaded_status(self):
        return {i: item.status for i, item in enumerate(self.items)}

    def __repr__(self):

        # get the status for the items
        status = self.loaded_status

        s = f"WNDataset instance with {len(self.items)} images\n"
        s += f"Loaded: {list(status.values()).count('Loaded')} items\n"
        s += f"Empty: {list(status.values()).count('Empty')} items\n"
        s += f"Loading: {list(status.values()).count('Loading')} items\n"

        return s

    def __iter__(self):
        # Simplest way to create an iterator is to yield the results (generator)
        for i in range(len(self)):
            yield self[i]
