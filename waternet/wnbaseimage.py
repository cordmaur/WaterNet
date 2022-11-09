from pathlib import Path
import rasterio as rio
import matplotlib as mpl
import matplotlib.pyplot as plt
from .wnutils import parse_sat_name
from typing import Union, Optional, cast, Tuple
from collections.abc import Callable
from matplotlib.axes import Axes
import numpy as np
import gc
import torch


def mask_transform(arr: np.ndarray):
    arr[arr == 255] = 2
    return arr.astype("uint8")


class WNBaseImage:

    home_path: Path = Path("/home/jovyan")
    img_type = "Planetary"  # 'S2COR', 'MAJA', 'THEIA'

    def __init__(
        self,
        path: Union[Path, str],
        transform: Optional[Callable] = mask_transform,
        shape: Optional[tuple] = None,
    ):
        p = Path(path)
        p = WNBaseImage.home_path / p if not p.is_absolute() else p

        self.ds = rio.open(p)
        self.transform = transform
        self._shape = shape if shape is not None else self.ds.shape  # type: ignore
        self._arr: Optional[np.ndarray] = None

    def load_array(self):
        # if there is a transform, apply it before loading the array
        if self._arr is None:
            self._arr = self.ds.read(out_shape=self._shape)  # type: ignore
            self._arr = cast(np.ndarray, self._arr)

            if self.transform is not None:
                self._arr = self.transform(self._arr)

        return self._arr

    @property
    def array(self):
        if self._arr is None:
            return self.load_array()
        else:
            return self._arr

    @property
    def path(self):
        return Path(self.ds.files[0])  # type: ignore

    @property
    def shape(self):
        return self._shape  # type: ignore

    @property
    def name(self):
        return self.path.name

    @property
    def properties(self):
        return parse_sat_name(self.path, WNBaseImage.img_type)

    def as_cube(self, bands) -> np.ndarray:
        if self._arr is not None:
            return self._arr
        else:
            raise Exception("Array is not loaded yet.")

    @staticmethod
    def get_cmap():
        cmap = mpl.cm.get_cmap("Set1")  # type: ignore
        order = [2, 1, 5, 0, 3, 4, 6, 7, 8]
        my_cmap = mpl.colors.ListedColormap([cmap.colors[i] for i in order], N=9)  # type: ignore
        return my_cmap

    @staticmethod
    def _plot(
        arr: np.ndarray,
        bright: float = 2.0,
        figsize: Tuple[int, int] = (10, 10),
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        """
        Internal function to plot an arbitrary array as image. Channels are expected to be in the first dim.
        If 1 or 2 channels, it will plot as grayscale (and discard 2nd channel).
        If 3 or more channels, it will plot RGB, considering the first 3 channels.


        Args:
            arr (np.ndarray): array to be plotted with the format (channels, height, width)
            bright (float, optional): brightness to be applied before plotting. Defaults to 2.0.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 10).
            ax (Optional[Axes], optional): Matplotlib axis to plot the image (if None, a new figura is createrd).
            Defaults to None.

        Raises:
            Exception: If the array cannot be plotted an exception is raised.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # before doing anything, let's copy the array (and convert to numpy if necessary)
        if isinstance(arr, torch.Tensor):
            arr = np.array(arr.detach().numpy())
        else:
            arr = arr.copy()

        arr = np.array(arr)

        # depending on the number of channels, plot as grey or RGB
        if (arr.ndim == 2) or (arr.shape[0] == 1):  # type: ignore
            arr = arr.squeeze()  # type: ignore

        # if 3 channels, them put the channels in the third dimension for plotting
        elif (arr.ndim == 3) and (arr.shape[0] >= 3):
            arr = arr[:3]
            arr = arr.transpose(1, 2, 0)

        elif (arr.ndim == 3) and (arr.shape[0] == 2):
            arr = arr[0].squeeze()

        else:
            raise Exception(f"Shape {arr.shape} cannot be plotted.")

        # Apply brightness gain only if it is float (not integers, because of masks)
        if not issubclass(arr.dtype.type, np.integer):
            arr = arr * bright
            arr[arr > 1] = 1
            arr[arr < 0] = 0
        elif "cmap" not in kwargs:
            # in this case, define a colormap
            cmap = WNBaseImage.get_cmap()
            kwargs["cmap"] = cmap
            kwargs["vmin"] = 0
            kwargs["vmax"] = cmap.N  # type: ignore

        ax.imshow(arr, **kwargs)

    def plot(
        self,
        ax: Optional[Axes] = None,
        downfactor: int = 10,
        figsize: Tuple[int, int] = (10, 10),
        bright: float = 2.0,
        **kwargs,
    ):
        """Plot the internal image. To save memory, it will be plotted in smaller resolution.

        Args:
            ax (Optional[Axes], optional): Matplotlib axis to plot the image (if None, a new figura is createrd).. Defaults to None.
            downfactor (int, optional): The factor used to divide the resolution. Defaults to 10.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 10).
            bright (float, optional): Brightness to be applied before plotting. Defaults to 2.0.
        """

        # make a 'smaller' view, and pass a copy
        arr = self.array[..., ::downfactor, ::downfactor].copy()
        WNBaseImage._plot(arr=arr, ax=ax, figsize=figsize, bright=bright, **kwargs)

    @property
    def loaded(self):
        return self._arr is not None

    def clear(self):
        del self._arr
        self._arr = None
        gc.collect()

    def __len__(self):
        return self.ds.count  # type: ignore

    def __repr__(self):
        s = f"WNBaseImage: array {'not ' if self._arr is None else ''}loaded in memory\n"
        return s
