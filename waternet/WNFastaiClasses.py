import sys
sys.path.append('D:\\Projects\\fastai')
sys.path.reverse()

# from fastai import *
# from fastai.data_block import *
from fastai.vision import *


class MSImage(Image):
    """
    This class is used by MSSegmentationItemList. It displays correctly a MS image using the first 3 bands
    """

    def _repr_image_format(self, format_str):
        with BytesIO() as str_buffer:
            # Use the first (1 or 3) channels to display the image
            chnls = 3 if self.px.shape[0] >= 3 else 1

            plt.imsave(str_buffer, image2np(self.px[0:chnls, :, :]), format=format_str)
            return str_buffer.getvalue()

    def show(self, ax: plt.Axes = None, figsize: tuple = (3, 3), title: Optional[str] = None, hide_axis: bool = True,
             cmap: str = None, y: Any = None, bright=1., **kwargs):
        # when displaying a MSImage we actually create a "fake" Image with 1 or 3 channels only and call the original SHOW
        chnls = 3 if self.px.shape[0] >= 3 else 1
        img = Image(self.px[0:chnls, :, :]*bright)
        img.show(ax, figsize, title, hide_axis, cmap, y, **kwargs)


class MSSegmentationLabelList(SegmentationLabelList):

    def open(self, fn):
        item = np.load(fn).astype('float32')
        # item = torch.load(fn).astype('float32')
        return ImageSegment(torch.tensor(item[np.newaxis, ...]))


#         return ImageSegment(torch.tensor(np.expand_dims(item, 0)))


class MSSegmentationItemList(SegmentationItemList):
    _label_cls = MSSegmentationLabelList

    def open(self, fn):
        item = np.load(fn).astype('float32')
        #         pdb.set_trace()
        # item = torch.load(fn).astype('float32')
        #         print (f'Passing SegmentationItemList {fn} shape: {item.shape}')
        #         return MSImage(item)
        return MSImage(torch.tensor(item))

    def reconstruct(self, t: Tensor): return MSImage(t.float().clamp(min=0, max=1))


def get_lbl_fn(img_fn: Path):
    lbl_path = img_fn.parent.parent / 'labels'
    lbl_name = img_fn.name.replace('n_mndwin_ndwin_B11B2', 'water_mask')
    return (lbl_path / lbl_name).with_suffix('.torch')