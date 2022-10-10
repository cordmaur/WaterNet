from pathlib import Path
import rasterio as rio
import matplotlib.pyplot as plt
from .wnutils import parse_sat_name
import gc

class WNMask:
    
    home_path = Path('/home/jovyan')
    img_type = 'Planetary'  # 'S2COR', 'MAJA', 'THEIA'
    
    def __init__(self, path: str):
        p = Path(path)
        p = WNMask.home_path/p if not p.is_absolute() else p
        
        self.ds = rio.open(p)
        self.arr_ = None
    
    @property
    def mask(self):
        # if not loaded, load the mask
        if self.arr_ is None:
            self.arr_ = self.ds.read().squeeze()
            self.arr_[self.arr_ == self.ds.nodata] = 2
        
        return self.arr_
    
    @property
    def path(self): return Path(self.ds.files[0])

    @property
    def shape(self): return self.ds.shape

    @property
    def properties(self): return parse_sat_name(self.path, WNMask.img_type)
    
    def plot(self, ax=None, downfactor:int=10, figsize=(10, 10)):
        
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        ax.imshow(self.mask[::downfactor, ::downfactor])
    
    def clear(self):
        del self.arr_
        self.arr_ = None
        gc.collect()
        
