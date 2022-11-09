from pystac_client import Client
from datetime import datetime, timedelta
import planetary_computer as pc
import rasterio as rio
from patchify import patchify
from pystac import item
from typing import Iterable, Tuple

import matplotlib.pyplot as plt

import numpy as np
import gc

from .wnbaseimage import WNBaseImage

# from typeguard import typechecked


class WNStacImage(WNBaseImage):
    """
    WNStacImage is a high level representation of a Sentinel-2 image originally stored in the PlanetaryComputer.
    It loads the bands on demand and perform other functions as bands math and tiling (using patchify)
    """

    # @typechecked
    def __init__(self, stac_item: item.Item, shape: Tuple[int, int]):
        """
        Create an instance that represents one single Sentinel2 image, based on the corresponding stac item.
        The constructors .from_tile can also be used.
        """
        self.stac_item = stac_item
        self.loaded_bands = {}
        self._shape = shape
        self._patches = None

    @staticmethod
    def search_catalog(query: dict, date_range: str, collection="sentinel-2-l2a"):
        """
        Search the MSPC catalog for the specific GEOJson Query and datetime range
        query example: {"s2:mgrs_tile": {"eq": tile} }
        date_range example: '2022-08-10/2022-09-01'
        """
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["sentinel-2-l2a"], query=query, datetime=date_range
        )

        return search.get_all_items().items

    @staticmethod
    def search_tile(tile: str, date_range: str):
        """
        Search for a tile within a date range.
        Date_range example: '2022-08-10/2022-09-01', or '2022-08'
        """
        # create the query to fetch the specific tile
        query = {"s2:mgrs_tile": {"eq": tile}}

        return WNStacImage.search_catalog(query, date_range)

    @classmethod
    def from_tile(cls, tile: str, str_date: str, shape: Tuple = (5490, 5490)):
        """
        Return the image that corresponds to the tile and the exact datetime.
        Considering revisit time, it will search for 10 days window and then get the closest date.
        Tile has 5 characters (e.g., '22KGF').
        Date has the format: 2022-08-10
        """

        # fetch the images that corresponds to the query (hopefully just one)
        imgs = WNStacImage.search_tile(tile=tile, date_range=str_date)
        assert len(imgs) == 1

        return cls(imgs[0], shape)

    def load_band(self, band: str):
        """
        Load one band into the memory.
        """
        if band not in self.stac_item.assets:
            raise Exception(f"Band {band} not available in assets")

        else:
            href = pc.sign(self.stac_item.assets[band].href)

            # read from the cloud with the correct shape
            b = rio.open(href).read(out_shape=self.shape).squeeze()  # type: ignore
            self.loaded_bands[band] = b.astype("float32") / 10000

    def load_bands(self, bands: Iterable[str]):
        """
        Load a list of bands into the memory
        """
        for band in bands:
            self.load_band(band)

    def get_band(self, band: str):
        """
        Return the band as array
        """

        if band not in self.loaded_bands:
            self.load_band(band)

        return self.loaded_bands[band]

    def as_cube(self, bands: Iterable[str]):
        """
        Return the bands as a cube, with the layer in last axis
        """

        bands_list = [self.get_band(band) for band in bands]

        return np.stack(bands_list, axis=0)

    @property
    def loaded(self) -> bool:
        return len(self.loaded_bands) > 0

    @property
    def array(self) -> np.ndarray:
        """
        The array property of the WNStacImage returns all loaded bands as a cube.
        If one want to return just specific bands, or force it to load bands, the .as_cube should be used instead.
        """
        if len(self.loaded_bands) == 0:
            raise Exception(
                f"No loaded bands in WNStacImage. Call load_bands or as_cube instead."
            )

        else:
            return self.as_cube(bands=self.loaded_bands.keys())

    @property
    def name(self):
        return self.stac_item.id

    def plot(
        self,
        bands: Iterable[str] = ["B04", "B03", "B02"],
        bright: float = 2.0,
        ax=None,
        figsize=(10, 10),
        downfactor=10,
        **kwargs,
    ):
        """
        Plot the image considering the bands in R, G and B positions
        If an axis is passed, plot the image inside it, otherwise, create a new axis
        """
        # first, get the array in lower resolution
        arr = self.as_cube(bands)[..., ::downfactor, ::downfactor]
        WNBaseImage._plot(arr=arr, ax=ax, figsize=figsize, bright=bright, **kwargs)

    def __getitem__(self, bands: list):

        if not isinstance(bands, (list, tuple)):
            bands = [bands]

        return [self.stac_item.assets[band] for band in bands]

    def __len__(self):
        return len(self.loaded_bands)

    def __repr__(self):
        s = f"Img: {self.stac_item.id}\n"
        s += f"Loaded bands: {list(self.loaded_bands.keys())}"
        return s

    def clear(self):

        for band in list(self.loaded_bands.keys()):
            del self.loaded_bands[band]
        gc.collect()
