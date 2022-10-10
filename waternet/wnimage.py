import pystac
from pystac_client import Client
from datetime import datetime, timedelta
import planetary_computer as pc
import rasterio as rio
from patchify import patchify

import matplotlib.pyplot as plt

import numpy as np
import gc

class WNImage:
    '''
    WNImage is a high level representation of a Sentinel-2 image originally stored in the PlanetaryComputer.
    It loads the bands on demand and perform other functions as bands math and tiling (using patchify)
    '''
    
    def __init__(self, stac_item: pystac.Item, shape: tuple):
        '''
        Create an instance that represents one single Sentinel2 image, based on the corresponding stac item.
        The constructors .from_tile can also be used.
        '''
        self.stac_item = pc.sign(stac_item)
        self.loaded_bands = {}
        self.shape = shape
        self._patches = None
        
        
    @staticmethod
    def search_catalog(query: dict, date_range: str, collection='sentinel-2-l2a'):
        '''
        Search the MSPC catalog for the specific GEOJson Query and datetime range
        query example: {"s2:mgrs_tile": {"eq": tile} }
        date_range example: '2022-08-10/2022-09-01'
        '''
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(collections=["sentinel-2-l2a"], query=query, datetime=date_range)
        
        return search.get_all_items().items
        
    @staticmethod
    def search_tile(tile: str, date_range: str):
        '''
        Search for a tile within a date range.
        Date_range example: '2022-08-10/2022-09-01', or '2022-08'
        '''
        # create the query to fetch the specific tile
        query = {'s2:mgrs_tile': {'eq': tile}}
        
        return WNImage.search_catalog(query, date_range)
                
    @classmethod
    def from_tile(cls, tile: str, str_date: str, shape: tuple=(5490, 5490)):
        '''
        Return the image that corresponds to the tile and the exact datetime. 
        Considering revisit time, it will search for 10 days window and then get the closest date.
        Tile has 5 characters (e.g., '22KGF').
        Date has the format: 2022-08-10
        '''
        
        # fetch the images that corresponds to the query (hopefully just one)
        imgs = WNImage.search_tile(tile=tile, date_range=str_date)
        assert len(imgs) == 1
        
        return cls(imgs[0], shape)
    
    @property
    def patch_size(self):
        '''
        Get the size of each single patch (channels in last dimension)
        '''
        if self._patches is None:
            print(f'No patches have been created yet. Create them with object.patchify.')
        else:
            return self._patches.shape[-3:]
    
    @property
    def patches(self):
        '''
        Return the an array with the patches. Each patch is indexed in the first dimension
        '''
        if self._patches is not None:
            return self._patches.reshape((-1,)+self.patch_size)
            
    def load_band(self, band: str):
        '''
        Load one band into the memory.
        '''
        if band not in self.stac_item.assets:
            raise Exception(f'Band {band} not available in assets')
        
        else:
            self.loaded_bands[band] = rio.open(self.stac_item.assets[band].href).read(out_shape=self.shape).squeeze()
              
    def load_bands(self, bands: list):
        '''
        Load a list of bands into the memory
        '''
        for band in bands:
            self.load_band(band)
    
    def get_band(self, band: str):
        '''
        Return the band as array
        '''
        
        if band not in self.loaded_bands:
            self.load_band(band)
            
        return self.loaded_bands[band]
      
    def as_cube(self, bands: list):
        '''
        Return the bands as a cube, with the layer in last axis
        '''

        bands_list = [self.get_band(band) for band in bands]
        
        return np.stack(bands_list, axis=-1)/10000
    
    def patchify(self, bands: list, patch_size: tuple, step: int):
        '''
        Create the patches for the image.
        The result will be stored in self.patches
        '''
        
        # get the image cube
        cube = self.as_cube(bands=bands)
        
        print()
        # create the patches accordingly
        self._patches = patchify(cube, patch_size=patch_size + (len(bands),), step=step)
        
    def plot(self, bands: list=['B04', 'B03', 'B02'], bright: float=2., ax=None, figsize=(10, 10), downfactor=10):
        '''
        Plot the image considering the bands in R, G and B positions
        If an axis is passed, plot the image inside it, otherwise, create a new axis
        '''
        
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        cube = self.as_cube(bands) * bright
        cube[cube>1] = 1
        
        ax.imshow(cube[::downfactor, ::downfactor])
        
    def plot_patches(self, bands: list=['B04', 'B03', 'B02'], bright: float=2., ax=None, figsize=(10, 10), downfactor=10):
        pass
         
    def __getitem__(self, bands: list):
        
        if not isinstance(bands, (list, tuple)):
            bands = [bands]
        
        return [self.stac_item.assets[band] for band in bands]
        
    def __len__(self):
        if self._patches is not None:
            return self.patches.shape[0]
        else:
            return 0
    
    def __repr__(self):
        s = f'Img: {self.stac_item.id}\n'
        s += f'Loaded bands: {list(self.loaded_bands.keys())}'
        return s
    
    def clear(self):
        
        for band in list(self.loaded_bands.keys()):
            del self.loaded_bands[band]
        gc.collect()
        