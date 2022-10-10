from pathlib import Path
import rasterio as rio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .wnutils import parse_sat_name
from .wnimage import WNImage
from .wnmask import WNMask

from patchify import patchify, unpatchify
from typing import Union
import time
from tqdm.notebook import tqdm

from concurrent.futures import ThreadPoolExecutor

import gc


class WNSegmentationItem:
    
    
    patch_size = (512, 512)
    
    
    def __init__(self, img: WNImage, mask: WNMask):
        self.img = img
        self.mask = mask
        self.mask_patches = self.img_patches = None
        self.calc_len()
        
        self.status = 'Empty'
        
    @classmethod
    def from_mask(cls, mask_path: str):
        
        # open the mask (do not load into memory)
        mask = WNMask(mask_path)
        
        # get the mask properties to find correct image
        mask_prop = mask.properties
        dt = '-'.join([mask_prop['year'], mask_prop['month'], mask_prop['day']])
        img = WNImage.from_tile(mask_prop['tile'], dt, mask.shape)
        
        return cls(img, mask)
    
    @property
    def shape(self): return self.mask.shape
    
    def calc_len(self):
        '''
        Use a mockup to guess the number of patches
        '''
        
        arr = np.zeros(self.mask.shape)
        patches = patchify (arr, WNSegmentationItem.patch_size, WNSegmentationItem.patch_size[0])
        patches = patches.reshape((-1,) + WNSegmentationItem.patch_size)
        
        self._len = len(patches)
    
    def patchify(self, bands: list):
        
        # set the status to loading
        self.status = 'Loading'
        
        # create the patches for the images
        img_cube = self.img.as_cube(bands)
        
        chnls = len(bands)
        patch_size = WNSegmentationItem.patch_size + (chnls,)
        
        self.img_patches = patchify(img_cube, patch_size, patch_size[0])
        self.img_patches = self.img_patches.reshape((-1,) + patch_size)
        
        # create the patches for the mask
        self.mask_patches = patchify(self.mask.mask, WNSegmentationItem.patch_size, WNSegmentationItem.patch_size[0])
        self.mask_patches =self.mask_patches.reshape((-1,) + WNSegmentationItem.patch_size)
        
        # set the loaded status
        self.status = 'Loaded'
        
    def plot(self, figsize:tuple=(20, 10)):
        
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        self.img.plot(ax=ax[0])
        self.mask.plot(ax=ax[1])

    def plot_patch(self, idx, chnls:slice=slice(0, 3), ax=None, targ:bool=False, figsize:tuple=(10, 5), bright=2.):
        
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=figsize)
        
        # plot the image patch
        img_patch = self.img_patches[idx][...] * bright
        img_patch[img_patch > 1] = 1
        ax[0].imshow(img_patch[..., chnls])
        

        # plot the mask
        mask_patch = self.mask_patches[idx]
        ax[1].imshow(mask_patch)
        
    def decode_batch(self, b): return b
        
    def __len__(self): return self._len

    def __getitem__(self, idx):
        
        if self.status == 'Empty':
            raise Exception('No patches created. Use .patchify method first.')
        
        if self.status == 'Loading':
            while self.status == 'Loading':
                time.sleep(0.2)

        # Before passing the image patch, we will invert the channel position to comply with PyTorch
        return self.img_patches[idx].transpose((2, 0, 1)).astype('float32'), self.mask_patches[idx]

    def __repr__(self):
        return f'{self.img.stac_item.id}->{self.mask.path.name}'
    
    def clear(self):
        self.status = 'Empty'
        self.img.clear()
        self.mask.clear()
        
        
        del self.mask_patches
        del self.img_patches
        self.mask_patches = None
        self.img_patches = None
        gc.collect()
    
    
class WNDataSet:
    
    def __init__(self, items: [WNSegmentationItem]):
        '''
        This function should not be called directly. Call .from_masks instead.
        '''
        self.items = items
        self._bands = None
        
        # create a pool for executing concurrent tasks
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.processes = []
    
    @classmethod
    def from_masks(cls, masks_path: Union[str, Path], patch_size:tuple=(256, 256), pattern:str='*', out_shape:tuple=None, retries:int=3):
        '''
        Create a new WNDataset instance from the masks. It match the masks with image IDs for quick access to the MSPC.
        :out_shape: if None, will load everything in the resolution of the mask, otherwise, will force a shape
        :patch_size: size for the patches that will be created for training the model
        :retries: number of retries if pc api is not responding to perform the matching
        '''
        
        # get the masks
        path = Path(masks_path)
        assert path.exists()
        
        # create SegmentationItems for each mask
        masks = list(path.rglob(pattern))
        items = []
        
        for mask in tqdm(masks, desc='Matching imgs/masks', ):
            retry = retries
            
            while retry > 0:
                try:
                    item = WNSegmentationItem.from_mask(mask)
                    items.append(item)
                    retry = 0
                    
                except Exception as e:
                    print(e)
                    retry = retry - 1
        
        return cls(items)
        
    def __len__(self): return sum(map(len, self.items))

    @property
    def bands(self): return self._bands

    @bands.setter
    def bands(self, value: list): self._bands = value
    
    @staticmethod
    def finished_loading(future):
        print('Loaded sucessfully')
        
    def load_next(self, int_position):
        
        # set the next index
        next_position = int_position + 1 if int_position < len(self.items) - 1 else 0
            
        # if the next position is empty, load it in parallel
        if self.items[next_position].status == 'Empty':
            print(f'Loading image {next_position} in background')
            process = self.pool.submit(self.items[next_position].patchify, self.bands)
            process.add_done_callback(WNDataSet.finished_loading)
            self.processes.append(process)
    
    def clear_previous(self, int_position):
        
        # set the previous index
        previous_position = int_position -1 if int_position > 0 else len(self.items) - 1
        
        if self.items[previous_position].status == 'Loaded':
            print(f'Cleared image {previous_position}')
            self.items[previous_position].clear()
            
    def __getitem__(self, idx):

        if self._bands is None:
            raise Exception('Bands not set. Set desired bands first.')
            
        if idx >= len(self):
            raise Exception('Index out of bounds of the dataset')
        
        # first, find the item for this specific idx
        # in terms of int_position (which item) and rel_position (which patch within the item)
        breaks = np.cumsum(list(map(len, self.items)))
        int_position = (breaks <= idx).sum()
        breaks = np.insert(breaks, 0, 0)
        rel_position = idx - breaks[int_position]
        
        # with the int_position, patchify the corresponding segmentation_item (if needed)
        item = self.items[int_position]
        if item.status == 'Empty':
            item.patchify(self.bands)
        
        # then check for the next and previous images
        self.clear_previous(int_position)
        self.load_next(int_position)
            
        return item[rel_position]    
    
    def clear(self):
        for item in self.items:
            item.clear()

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
    
    # Criar um status para o SegItem
    # loading -> para evitar lançar 2 processos de loading ao mesmo tempo
    # loaded -> para limpar a memória, caso necessário
    # 