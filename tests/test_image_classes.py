from pathlib import Path
import numpy as np

import pytest

from waternet.wnbaseimage import WNBaseImage
from waternet.wnsegmantationitem import WNSegmentationItem
from waternet.wnstacimage import WNStacImage
from waternet.wnimagepatcher import WNImagePatcher


@pytest.fixture(scope="session")
def test_data():
    fixtures = {
        "mask_path": "d:/temp/22KEV/S2A_MSIL2A_20180123T133221_R081_T22KEV_watermask.tif",
        "stac_tile": "22KEV",
        "stac_date": "2022-08-25",
        "stac_name": "S2B_MSIL2A_20220825T133149_R081_T22KEV_20220826T055538",
        "stac_shape": (5490, 5490),
        "stac_bands": ["B01", "B09"],
    }
    return fixtures


@pytest.fixture(scope="session")
def stac_img(test_data):
    """Load a stac_img that will be used accross multiple tests."""

    stac_img = WNStacImage.from_tile(
        tile=test_data["stac_tile"],
        str_date=test_data["stac_date"],
        shape=test_data["stac_shape"],
    )

    assert stac_img.shape == test_data["stac_shape"]
    assert stac_img.name == test_data["stac_name"]

    return stac_img


class TestWNBaseImage:
    def test_baseimage(self, test_data):

        # taking an image as reference
        p = Path(test_data["mask_path"])

        # Opening a S2_Image and checking shape
        img = WNBaseImage(p, transform=None, shape=None)
        assert img.shape == (10980, 10980)

        # Opening a S2_Image with transformation and different shape and checking results
        img = WNBaseImage(
            p, transform=lambda arr: arr.astype("uint16"), shape=test_data["stac_shape"]
        )
        assert img.array.shape == (1, 5490, 5490)
        assert img.array.dtype == "uint16"

        assert img.name == test_data["mask_path"].split("/")[-1]
        # clearing memory
        img.clear()
        assert not img.loaded


class TestWNStacImage:
    def test_stacimage(self, test_data, stac_img):

        # Loading 60m bands (and checking if they will output as our shape)
        cube = stac_img.as_cube(bands=test_data["stac_bands"])
        assert cube.shape == (2,) + test_data["stac_shape"]
        assert stac_img.array.shape == (2,) + test_data["stac_shape"]

        assert stac_img.array.dtype == "float32"

        # clearing memory
        # stac_img.clear()
        # assert stac_img.loaded == False
        # assert len(stac_img.loaded_bands) == 0

    def test_reload(self, stac_img):
        print("testing reloading")
        print(stac_img)
        assert stac_img.loaded


class TestWNImagePatcher:
    @staticmethod
    def transform(arr):
        arr[arr == 255] = 2
        return arr.astype("uint8")

    def validate_patcher(self, input_img, bands=None):

        patcher = WNImagePatcher(img=input_img, patch_size=(512, 512), step=262)

        assert len(patcher) == 400

        # create the patches
        patcher.patchify(bands=bands)

        # test if patches have the correct size
        assert patcher.patch_size == (len(input_img), 512, 512)
        assert patcher.patches.shape == (len(patcher),) + patcher.patch_size  # type: ignore
        assert patcher[0].shape == patcher.patch_size

        # test if unpacthify is working correctly
        assert np.all(patcher.unpatchify() == patcher.img.array)

    def test_imagepatcher_baseimage(self, test_data):

        # Will test the patcher in a mask (quicker)
        p = Path(test_data["mask_path"])
        mask = WNBaseImage(
            p, transform=TestWNImagePatcher.transform, shape=(5490, 5490)
        )

        assert mask.array.max() == 2
        assert mask.array.min() == 0

        self.validate_patcher(mask, bands=None)

    def test_imagepatcher_stacimage(self, stac_img, test_data):

        # test if it is raising expection for non-divisible step
        with pytest.raises(Exception):
            WNImagePatcher(img=stac_img, patch_size=(512, 512), step=256)

        self.validate_patcher(stac_img, bands=test_data["stac_bands"])


class TestWNSegmentationItem:
    def test_segmentation_item(self, test_data, stac_img):
        # load a mask
        mask = WNBaseImage(
            test_data["mask_path"],
            transform=TestWNImagePatcher.transform,
            shape=test_data["stac_shape"],
        )

        # create the segmentation item directly from the contructor
        segitem = WNSegmentationItem(
            img=stac_img, mask=mask, patch_size=(512, 512), step=262
        )

        assert segitem.shape == test_data["stac_shape"]

        segitem.patchify(bands=test_data["stac_bands"])
        assert segitem.status == "Loaded"

        x = segitem[0]
        assert isinstance(x, tuple)
        assert x[0].shape == (len(stac_img), 512, 512)
        assert x[1].shape == (512, 512)
