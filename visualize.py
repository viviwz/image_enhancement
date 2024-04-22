import numpy as np
from pathlib import Path
from PIL import Image
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
import imageio.v3 as iio
from typing import Union, Iterable, Sequence

DEFAULT_DPI = 200

# Default colors (from names)
_default_colors = dict(
        red=(1,0,0),
        green=(0,1,0),
        blue=(0,0,1),
        magenta=(1,0,1),
        yellow=(1,1,0),
        cyan=(0,1,1),
        white=(1,1,1),
        black=(0,0,0)
    )

def rescale(im):
    # Convert to float32 and rescale per channel (to simplify display later)
    in_dims = im.ndim
    if in_dims <= 3:
        im = np.expand_dims(im, -1)

    im = im.astype(np.float32)
    for c in range(im.shape[-1]):
        im[..., c] = im[..., c] - np.percentile(im[..., ], 0.25)
        im[..., c] = im[..., c] / np.percentile(im[..., ], 99.75) 

    if in_dims <= 3:
        im = np.squeeze(im)

    return im

def display_image(x):
    # x_scaled = np.uint8(255 * (x - x.min()) / x.ptp())
    x_scaled = x
    # plt.imshow(x, cmap='gray')
    return Image.fromarray(x_scaled)
    return 
    
def display_slices(images):
    def _show(slice=(0, len(images)-1)):
        return display_image(images[slice,...])
    return interact(_show)

def create_figure(figsize=None, dpi=DEFAULT_DPI, **kwargs):
    return plt.figure(figsize=figsize, dpi=dpi, **kwargs)

def show_image(im, pos=None, axis=False, title=None, axes=None, cmap='gray', clip_percentile=None,
        fontdict={'fontsize':'small'}, **kwargs):
    """
    Helper function to show an image using pyplot.
    """
    if pos:
        if isinstance(pos, int):
            plt.subplot(pos)
        else:
            plt.subplot(*pos)
    if type(im) in [str, bytes, Path]:
        im = _load_image(im)
    if clip_percentile:
        kwargs = dict(kwargs)
        if not 'vmin' in kwargs:
            kwargs['vmin'] = np.percentile(im, clip_percentile)
        if not 'vmax' in kwargs:
            kwargs['vmax'] = np.percentile(im, 100-clip_percentile)
    if axes is not None:
        axes.imshow(im, cmap=cmap, **kwargs)
        if title:
            axes.set_title(title, fontdict=fontdict)
        if axis:
            axes.set_axis_on()
        elif axis is not None:
            axes.set_axis_off()
    else:
        plt.imshow(im, cmap=cmap, **kwargs)
        if title:
            plt.title(title, fontdict=fontdict)
        plt.axis(axis)

def show_orthogonal(im, axes, loc=None, projection=None, colors=None, show_slices=False, title=None):
    """
    Show orthogonal slices or projections.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_channels = im.shape[-1] if im.ndim > 3 else 1
    if colors is None:
        if n_channels == 1:
            colors = ('white',)
        elif n_channels == 2:
            colors = ('green', 'magenta')
        elif n_channels == 3:
            colors = ('red', 'green', 'blue')
        else:
            raise ValueError('Colors must be specified if the number of channels is > 3!')

    if axes is None:
        fig = create_figure()
        ax = fig.gca()
    else:
        ax = axes

    if loc is None and projection is None:
        loc = tuple(s//2 for s in im.shape)

    # Create axes for orthogonal views
    divider = make_axes_locatable(ax)
    ax_xz = divider.append_axes("bottom", size="50%", pad=0.1, sharex=ax)
    ax_yz = divider.append_axes("left", size="50%", pad=0.1, sharey=ax)

    # Generate slices or projections
    if projection:
        imxy = projection(im, axis=0)
        imxz = projection(im, axis=2)
        imyz = np.moveaxis(projection(im, axis=1), 0, 1)
    else:
        imxy = im[loc[0],...]
        imxz = im[:, loc[1],...]
        imyz = np.moveaxis(im[:, :, loc[2],:], 0, 1)


    show_image(create_rgb(imyz, colors),
                   axes=ax_yz, vmin=0, vmax=1)
    if show_slices:
        ax_yz.plot([loc[0], loc[0]], [0, im.shape[2]-1], 'w--', linewidth=1)
    ax_yz.set_xlabel('z')
    ax_yz.set_xticks([])
    ax_yz.set_ylabel('y', rotation=0, va='center', ha='right')
    ax_yz.set_yticks([])
    ax_yz.set_axis_on()

    show_image(create_rgb(imxy, colors),
                   axes=ax,
                   vmin=0, vmax=1)
    if show_slices:
        row = loc[1]
        col = loc[2]
        ax.plot([0, im.shape[1]-1], [row, row], 'w--', linewidth=1)
        ax.plot([col, col], [0, im.shape[2]-1], 'w--', linewidth=1)

    show_image(create_rgb(imxz, colors),
                   axes=ax_xz,
                   vmin=0, vmax=1)
    if show_slices:
        ax_xz.plot([0, im.shape[1]-1], [loc[0], loc[0]], 'w--', linewidth=1)
    ax_xz.set_xlabel('x')
    ax_xz.set_xticks([])
    ax_xz.set_ylabel('z', rotation=0, va='center', ha='right')
    ax_xz.set_yticks([])
    ax_xz.set_axis_on()

    plt.tight_layout()

    if title:
        ax.set_title(title)

def create_rgb(im: np.ndarray, 
        colors: Sequence[Iterable[Union[float,str]]], 
        vmin: Iterable[float]=None, vmax: Iterable[float]=None,
        axis: int=None):
    """
    Create an RGB image from individual channels and associated single-color LUTs.
    
    Channels can be a list of tuples, each containing a 2D image channel and 
    a color (represented as a tuple of 3 floats).
    Alternatively, channels can be an ndarray with the colors provided as a separate 
    iterable.
    
    Image channels can be uint8 (in which case the output is uint8) or float.
    """
    im_merged = 0.0
    is_int = False

    if axis is None:
        if len(colors) == im.shape[0]:
            axis = 0
        elif len(colors) == im.shape[-1]:
            axis = len(im.shape) - 1
        else:
            raise ValueError(f'Unable to match {len(colors)} channels for image with shape {im.shape}')

    if np.issubdtype(im.dtype, np.integer):
        is_int = True
        im = im.astype(np.float32)

    channels = np.split(im, im.shape[axis], axis=axis)
    rescale_int = is_int
    vmin = vmin if isinstance(vmin, Sequence) else [vmin] * len(colors)
    vmax = vmax if isinstance(vmax, Sequence) else [vmax] * len(colors)
    
    for im_channel, color, vmi, vma in zip(channels, colors, vmin, vmax):
        if isinstance(color, str):
            color_tuple = _default_colors.get(color.lower())
        else:
            color_tuple = _default_colors.get(color, color)
        if vmi and vma:
            im_channel = (im_channel - vmi) / (vma - vmi)
        elif vmi:
            im_channel = (im_channel - vmi)
        elif vma:
            im_channel = im_channel / vma
        else:
            rescale_int = False
        im_merged += np.atleast_3d(im_channel) * np.asarray(color_tuple).reshape((1, 1, 3))
    
    if is_int:
        if rescale_int:
            im_merged = im_merged * 255
        return np.clip(im_merged, 0, 255).astype(np.uint8)
    else:
        return np.clip(im_merged, 0, 1.0)
    
def _load_image(data, volume:bool = False, metadata:bool = False, **kwargs):

    if metadata:
        return _load_image(data, metadata=False, **kwargs), iio.immeta(data, **kwargs)
    else:
        im = iio.imread(data, **kwargs)
        if volume:
            return im
        else:
            # If we don't want a volume image, try to squeeze down to the minimum
            # This is partly due to blobs.gif having a shape (1, ?, ?, 3) when it should really be single-channel grayscale
            im = np.squeeze(im)
            if im.ndim == 3 and im.shape[2] == 3 and np.array_equal(im[..., 0], im[..., 1]) and np.array_equal(im[..., 0], im[..., 2]):
                return im[..., 0]
            return im
        