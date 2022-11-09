from unittest.mock import patch
from .wnbaseimage import WNBaseImage
from .wnstacimage import WNStacImage
from typing import Optional, Iterable, Union, Tuple
from patchify import patchify, unpatchify
from matplotlib.axes import Axes
import numpy as np
import gc


class WNImagePatcher:
    def __init__(
        self,
        img: Union[WNBaseImage, WNStacImage],
        patch_size: Tuple[int, int] = (512, 512),
        step: int = 256,
    ):

        # Precalc the number of patches (and validates step value)
        self._len = WNImagePatcher.number_of_patches(
            shape=img.shape, patch_size2d=patch_size, step=step
        )

        self.img = img
        self._patch_size2d = patch_size
        self._step = step
        self._patches = None

    @staticmethod
    def validate_step(shape: Tuple[int, int], patch_size2d: Tuple[int, int], step: int):
        """
        This function will test if the patch_size and step will match the requirements to reconstruct the original image.
        """

        # will test sizes across both axis
        for axis in [0, 1]:
            width = shape[axis] - patch_size2d[axis]
            if width % step != 0:
                # if there is a problem, look for possible steps
                valid_steps = [
                    i for i in range(1, patch_size2d[axis]) if width % i == 0
                ]

                s = f"Invalid step value {step}. Consider using one of the following step values:\n"
                s += f"{valid_steps}"
                raise Exception(s)

        return True

    @staticmethod
    def number_of_patches(
        shape: Tuple[int, int], patch_size2d: Tuple[int, int], step: int
    ):
        """
        Use a mockup to guess the number of patches, considering image shape and patch_size
        """
        WNImagePatcher.validate_step(shape=shape, patch_size2d=patch_size2d, step=step)

        arr = np.empty(shape)
        patches = patchify(arr, patch_size2d, step)
        patches = patches.reshape((-1,) + patch_size2d)

        return len(patches)

    def patchify(self, bands: Optional[Iterable[str]] = None):
        """
        Create the patches for the image.
        The result will be stored in self.patches and can be acessed through __getitem__()

        Args:
            bands (Optional[Iterable[str]], optional): The bands to be used in the final patches. If none is provided, it will used loaded bands. Defaults to None.
        """

        # get the array to be patchified
        cube = self.img.array if bands is None else self.img.as_cube(bands)
        if cube is None:
            raise Exception("Array is empty and no bands are passed")

        # create the patches accordingly
        self._patches = patchify(
            image=cube,
            patch_size=(cube.shape[0],) + self._patch_size2d,
            step=self._step,
        )

        self._patchified_shape = self._patches.shape
        self._patches = self._patches.reshape((-1,) + self.patch_size)  # type: ignore

    def unpatchify(self):
        """
        Recreate the original scene
        """
        if self._patches is None:
            raise Exception("Cannot unpatchify. Patches not created")

        else:

            reconstruct = unpatchify(
                patches=self._patches.reshape(self._patchified_shape),
                imsize=self.img.array.shape,
            )

            if np.any(reconstruct != self.img.array):
                raise Exception("Reconstruction failed")

            return reconstruct

    def plot_patch(
        self,
        idx: int,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (4, 4),
        bright: float = 2.0,
        **kwargs,
    ):
        """
        Plot one patch.
        As we cannot recreate the patches for each visualization, it will use the first three dimensions.
        """
        # get the patch and slice the first 3 channels
        arr = self[idx][:3]
        WNBaseImage._plot(arr, bright=bright, ax=ax, figsize=figsize, **kwargs)

    @property
    def patch_size(self):
        """
        Get the size of each single patch (channels in first dimension)
        """
        if self._patches is None:
            print(
                f"No patches have been created yet. Create them with object.patchify."
            )
        else:
            return self._patches.shape[-3:]

    @property
    def patches(self):
        """
        Return the an array with the patches. Each patch is indexed in the first dimension
        """
        if self._patches is not None:
            return self._patches
        else:
            raise Exception(
                f"No patches created for {self.img.name}. Create them with .patchify."
            )

    def clear(self):
        self.img.clear()
        gc.collect()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """
        Return the patch in the given position.
        """

        return self.patches[idx]

    def __repr__(self):

        s = f"Patcher with {'no' if self._patches is None else len(self)} patches created."
        return s
