import os
from .vendored import colorconv, cm
import numpy as np
from vispy.color import get_colormap, get_colormaps, BaseColormap, Colormap

_matplotlib_list_file = os.path.join(
    os.path.dirname(__file__), 'matplotlib_cmaps.txt'
)
with open(_matplotlib_list_file) as fin:
    matplotlib_colormaps = [line.rstrip() for line in fin]


primary_color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
primary_colors = np.array(
    [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
    dtype=float,
)


simple_colormaps = {
    name: Colormap([[0.0, 0.0, 0.0], color])
    for name, color in zip(primary_color_names, primary_colors)
}


def _all_rgb():
    """Return all 256**3 valid rgb tuples."""
    base = np.arange(256, dtype=np.uint8)
    r, g, b = np.meshgrid(base, base, base, indexing='ij')
    return np.stack((r, g, b), axis=-1).reshape((-1, 3))


# obtained with colorconv.rgb2luv(_all_rgb().reshape((-1, 256, 3)))
LUVMIN = np.array([0.0, -83.07790815, -134.09790293])
LUVMAX = np.array([100.0, 175.01447356, 107.39905336])
LUVRNG = LUVMAX - LUVMIN

# obtained with colorconv.rgb2lab(_all_rgb().reshape((-1, 256, 3)))
LABMIN = np.array([0.0, -86.18302974, -107.85730021])
LABMAX = np.array([100.0, 98.23305386, 94.47812228])
LABRNG = LABMAX - LABMIN


def _validate_rgb(colors, *, tolerance=0.0):
    """Return the subset of colors that is in [0, 1] for all channels.

    Parameters
    ----------
    colors : array of float, shape (N, 3)
        Input colors in RGB space.

    Other Parameters
    ----------------
    tolerance : float, optional
        Values outside of the range by less than ``tolerance`` are allowed and
        clipped to be within the range.

    Returns
    -------
    filtered_colors : array of float, shape (M, 3), M <= N
        The subset of colors that are in valid RGB space.

    Examples
    --------
    >>> colors = np.array([[  0. , 1.,  1.  ],
    ...                    [  1.1, 0., -0.03],
    ...                    [  1.2, 1.,  0.5 ]])
    >>> _validate_rgb(colors)
    array([[0., 1., 1.]])
    >>> _validate_rgb(colors, tolerance=0.15)
    array([[0., 1., 1.],
           [1., 0., 0.]])
    """
    lo = 0 - tolerance
    hi = 1 + tolerance
    valid = np.all((colors > lo) & (colors < hi), axis=1)
    filtered_colors = np.clip(colors[valid], 0, 1)
    return filtered_colors


def _low_discrepancy_image(image, seed=0.5):
    """Generate a 1d low discrepancy sequence of coordinates.

    Parameters
    ----------
    labels : array of int
        A set of labels or label image.
    seed : float
        The seed from which to start the quasirandom sequence.

    Returns
    -------
    image_out : array of float
        The set of ``labels`` remapped to [0, 1] quasirandomly.

    """
    phi = 1.6180339887498948482
    image_out = (seed + image / phi) % 1
    # Clipping slightly above 0 and below 1 is necessary to ensure that the
    # labels do not get mapped to 0 which is represented by the background
    # and is transparent
    return np.clip(image_out, 0.00001, 1.0 - 0.00001)


def _low_discrepancy(dim, n, seed=0.5):
    """Generate a 1d, 2d, or 3d low discrepancy sequence of coordinates.

    Parameters
    ----------
    dim : one of {1, 2, 3}
        The dimensionality of the sequence.
    n : int
        How many points to generate.
    seed : float or array of float, shape (dim,)
        The seed from which to start the quasirandom sequence.

    Returns
    -------
    pts : array of float, shape (n, dim)
        The sampled points.

    References
    ----------
    ..[1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/  # noqa: E501
    """
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    seed = np.broadcast_to(seed, (1, dim))
    phi = np.array([phi1, phi2, phi3])
    g = 1 / phi
    n = np.reshape(np.arange(n), (n, 1))
    pts = (seed + (n * g[:dim])) % 1
    return pts


def _color_random(n, *, colorspace='lab', tolerance=0.0, seed=0.5):
    """Generate n random RGB colors uniformly from LAB or LUV space.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    colorspace : str, one of {'lab', 'luv', 'rgb'}
        The colorspace from which to get random colors.
    tolerance : float
        How much margin to allow for out-of-range RGB values (these are
        clipped to be in-range).
    seed : float or array of float, shape (3,)
        Value from which to start the quasirandom sequence.

    Returns
    -------
    rgb : array of float, shape (n, 3)
        RGB colors chosen uniformly at random from given colorspace.
    """
    # factor = 6  # about 1/5 of random LUV tuples are inside the space
    # expand_factor = 2
    # rgb = np.zeros((0, 3))
    # while len(rgb) < n:
    #     random = _low_discrepancy(3, n * factor, seed=seed)
    #     if colorspace == 'luv':
    #         raw_rgb = colorconv.luv2rgb(random * LUVRNG + LUVMIN)
    #     elif colorspace == 'rgb':
    #         raw_rgb = random
    #     else:  # 'lab' by default
    #         raw_rgb = colorconv.lab2rgb(random * LABRNG + LABMIN)
    #     rgb = _validate_rgb(raw_rgb, tolerance=tolerance)
    #     factor *= expand_factor
    # return rgb[:n]
    return GLASBEY[:n]


def label_colormap(num_colors=256, seed=0.5):
    """Produce a colormap suitable for use with a given label set.

    Parameters
    ----------
    num_colors : int, optional
        Number of unique colors to use. Default used if not given.
    seed : float or array of float, length 3
        The seed for the random color generator.

    Returns
    -------
    cmap : vispy.color.Colormap
        A colormap for use with labels are remapped to [0, 1].

    Notes
    -----
    0 always maps to fully transparent.
    """
    # Starting the control points slightly above 0 and below 1 is necessary
    # to ensure that the background pixel 0 is transparent
    midpoints = np.linspace(0.00001, 1 - 0.00001, num_colors - 1)
    control_points = np.concatenate(([0], midpoints, [1.0]))
    # make sure to add an alpha channel to the colors
    colors = np.concatenate(
        (_color_random(num_colors, seed=seed), np.full((num_colors, 1), 1)),
        axis=1,
    )
    colors[0, :] = 0  # ensure alpha is 0 for label 0
    cmap = Colormap(
        colors=colors, controls=control_points, interpolation='zero'
    )
    return cmap


def vispy_or_mpl_colormap(name):
    """Try to get a colormap from vispy, or convert an mpl one to vispy format.

    Parameters
    ----------
    name : str
        The name of the colormap.

    Returns
    -------
    cmap : vispy.color.Colormap
        The found colormap.

    Raises
    ------
    KeyError
        If no colormap with that name is found within vispy or matplotlib.
    """
    vispy_cmaps = get_colormaps()
    if name in vispy_cmaps:
        cmap = get_colormap(name)
    else:
        try:
            mpl_cmap = getattr(cm, name)
        except AttributeError:
            raise KeyError(
                f'Colormap "{name}" not found in either vispy '
                'or matplotlib.'
            )
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        cmap = Colormap(mpl_colors)
    return cmap


# Fire and Grays are two colormaps that work well for
# translucent and additive volume rendering - add
# them to best_3d_colormaps, append them to
# all the existing colormaps


class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """

    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack(
                [np.power(t, 0.5), t, t * t, np.maximum(0, t * 1.05 - 0.05)]
            ).astype(np.float32)
        else:
            return np.array(
                [np.power(t, 0.5), t, t * t, np.maximum(0, t * 1.05 - 0.05)],
                dtype=np.float32,
            )


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.5);
    }
    """

    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack([t, t, t, t * 0.5]).astype(np.float32)
        else:
            return np.array([t, t, t, t * 0.5], dtype=np.float32)


colormaps_3D = {"fire": TransFire(), "gray_trans": TransGrays()}
colormaps_3D = {k: v for k, v in sorted(colormaps_3D.items())}


# A dictionary mapping names to VisPy colormap objects
ALL_COLORMAPS = {k: vispy_or_mpl_colormap(k) for k in matplotlib_colormaps}
ALL_COLORMAPS.update(simple_colormaps)
ALL_COLORMAPS.update(colormaps_3D)

# ... sorted alphabetically by name
AVAILABLE_COLORMAPS = {k: v for k, v in sorted(ALL_COLORMAPS.items())}

# curated colormap sets
# these are selected to look good or at least reasonable when using additive
# blending of multiple channels.
MAGENTA_GREEN = ['magenta', 'green']
RGB = ['red', 'green', 'blue']
CYMRGB = ['cyan', 'yellow', 'magenta', 'red', 'green', 'blue']

GLASBEY = [[1.        , 1.        , 1.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 0.2       ],
           [1.        , 0.        , 0.71372549],
           [0.        , 0.3254902 , 0.        ],
           [1.        , 0.82745098, 0.        ],
           [0.        , 0.62352941, 1.        ],
           [0.60392157, 0.30196078, 0.25882353],
           [0.        , 1.        , 0.74509804],
           [0.47058824, 0.24705882, 0.75686275],
           [0.12156863, 0.58823529, 0.59607843],
           [1.        , 0.6745098 , 0.99215686],
           [0.69411765, 0.8       , 0.44313725],
           [0.94509804, 0.03137255, 0.36078431],
           [0.99607843, 0.56078431, 0.25882353],
           [0.86666667, 0.        , 1.        ],
           [0.1254902 , 0.10196078, 0.00392157],
           [0.44705882, 0.        , 0.33333333],
           [0.4627451 , 0.42352941, 0.58431373],
           [0.00784314, 0.67843137, 0.14117647],
           [0.78431373, 1.        , 0.        ],
           [0.53333333, 0.42352941, 0.        ],
           [1.        , 0.71764706, 0.62352941],
           [0.52156863, 0.52156863, 0.40392157],
           [0.63137255, 0.01176471, 0.        ],
           [0.07843137, 0.97647059, 1.        ],
           [0.        , 0.27843137, 0.61960784],
           [0.8627451 , 0.36862745, 0.57647059],
           [0.57647059, 0.83137255, 1.        ],
           [0.        , 0.29803922, 1.        ],
           [0.        , 0.25882353, 0.31372549],
           [0.22352941, 0.65490196, 0.41568627],
           [0.93333333, 0.43921569, 0.99607843],
           [0.        , 0.        , 0.39215686],
           [0.67058824, 0.96078431, 0.8       ],
           [0.63137255, 0.57254902, 1.        ],
           [0.64313725, 1.        , 0.45098039],
           [1.        , 0.80784314, 0.44313725],
           [0.27843137, 0.        , 0.08235294],
           [0.83137255, 0.67843137, 0.77254902],
           [0.98431373, 0.4627451 , 0.43529412],
           [0.67058824, 0.7372549 , 0.        ],
           [0.45882353, 0.        , 0.84313725],
           [0.65098039, 0.        , 0.60392157],
           [0.        , 0.45098039, 0.99607843],
           [0.64705882, 0.36470588, 0.68235294],
           [0.38431373, 0.51764706, 0.00784314],
           [0.        , 0.4745098 , 0.65882353],
           [0.        , 1.        , 0.51372549],
           [0.3372549 , 0.20784314, 0.        ],
           [0.62352941, 0.        , 0.24705882],
           [0.25882353, 0.17647059, 0.25882353],
           [1.        , 0.94901961, 0.73333333],
           [0.        , 0.36470588, 0.2627451 ],
           [0.98823529, 1.        , 0.48627451],
           [0.62352941, 0.74901961, 0.72941176],
           [0.65490196, 0.32941176, 0.0745098 ],
           [0.29019608, 0.15294118, 0.42352941],
           [0.        , 0.0627451 , 0.65098039],
           [0.56862745, 0.30588235, 0.42745098],
           [0.81176471, 0.58431373, 0.        ],
           [0.76470588, 0.73333333, 1.        ],
           [0.99215686, 0.26666667, 0.25098039],
           [0.25882353, 0.30588235, 0.1254902 ],
           [0.41568627, 0.00392157, 0.        ],
           [0.70980392, 0.51372549, 0.32941176],
           [0.51764706, 0.91372549, 0.57647059],
           [0.37647059, 0.85098039, 0.        ],
           [1.        , 0.43529412, 0.82745098],
           [0.4       , 0.29411765, 0.24705882],
           [0.99607843, 0.39215686, 0.        ],
           [0.89411765, 0.01176471, 0.49803922],
           [0.06666667, 0.78039216, 0.68235294],
           [0.82352941, 0.50588235, 0.54509804],
           [0.35686275, 0.4627451 , 0.48627451],
           [0.1254902 , 0.23137255, 0.41568627],
           [0.70588235, 0.32941176, 1.        ],
           [0.88627451, 0.03137255, 0.82352941],
           [0.        , 0.00392157, 0.07843137],
           [0.36470588, 0.51764706, 0.26666667],
           [0.65098039, 0.98039216, 1.        ],
           [0.38039216, 0.48235294, 0.78823529],
           [0.38431373, 0.        , 0.47843137],
           [0.49411765, 0.74509804, 0.22745098],
           [0.        , 0.23529412, 0.71764706],
           [1.        , 0.99215686, 0.        ],
           [0.02745098, 0.77254902, 0.88627451],
           [0.70588235, 0.65490196, 0.22352941],
           [0.58039216, 0.72941176, 0.54117647],
           [0.8       , 0.73333333, 0.62745098],
           [0.21568627, 0.        , 0.19215686],
           [0.        , 0.15686275, 0.00392157],
           [0.58823529, 0.47843137, 0.50588235],
           [0.15294118, 0.53333333, 0.14901961],
           [0.80784314, 0.50980392, 0.70588235],
           [0.58823529, 0.64313725, 0.76862745],
           [0.70588235, 0.1254902 , 0.50196078],
           [0.43137255, 0.3372549 , 0.70588235],
           [0.57647059, 0.        , 0.7254902 ],
           [0.78039216, 0.18823529, 0.23921569],
           [0.45098039, 0.4       , 1.        ],
           [0.05882353, 0.73333333, 0.99215686],
           [0.6745098 , 0.64313725, 0.39215686],
           [0.71372549, 0.45882353, 0.98039216],
           [0.84705882, 0.8627451 , 0.99607843],
           [0.34117647, 0.55294118, 0.44313725],
           [0.84705882, 0.33333333, 0.13333333],
           [0.        , 0.76862745, 0.40392157],
           [0.95294118, 0.64705882, 0.41176471],
           [0.84705882, 1.        , 0.71372549],
           [0.00392157, 0.09411765, 0.85882353],
           [0.20392157, 0.25882353, 0.21176471],
           [1.        , 0.60392157, 0.        ],
           [0.34117647, 0.37254902, 0.00392157],
           [0.77647059, 0.94509804, 0.30980392],
           [1.        , 0.37254902, 0.52156863],
           [0.48235294, 0.6745098 , 0.94117647],
           [0.47058824, 0.39215686, 0.19215686],
           [0.63529412, 0.52156863, 0.8       ],
           [0.41176471, 1.        , 0.8627451 ],
           [0.77647059, 0.32156863, 0.39215686],
           [0.4745098 , 0.10196078, 0.25098039],
           [0.        , 0.93333333, 0.2745098 ],
           [0.90588235, 0.81176471, 0.27058824],
           [0.85098039, 0.50196078, 0.91372549],
           [1.        , 0.82745098, 0.81960784],
           [0.81960784, 1.        , 0.55294118],
           [0.14117647, 0.        , 0.01176471],
           [0.34117647, 0.63921569, 0.75686275],
           [0.82745098, 0.90588235, 0.78823529],
           [0.79607843, 0.43529412, 0.30980392],
           [0.24313725, 0.09411765, 0.        ],
           [0.        , 0.45882353, 0.8745098 ],
           [0.43921569, 0.69019608, 0.34509804],
           [0.81960784, 0.09411765, 0.        ],
           [0.        , 0.11764706, 0.41960784],
           [0.41176471, 0.78431373, 0.77254902],
           [1.        , 0.79607843, 1.        ],
           [0.91372549, 0.76078431, 0.5372549 ],
           [0.74901961, 0.50588235, 0.18039216],
           [0.27058824, 0.16470588, 0.56862745],
           [0.67058824, 0.29803922, 0.76078431],
           [0.05490196, 0.45882353, 0.23921569],
           [0.        , 0.11764706, 0.09803922],
           [0.4627451 , 0.28627451, 0.49803922],
           [1.        , 0.6627451 , 0.78431373],
           [0.36862745, 0.21568627, 0.85098039],
           [0.93333333, 0.90196078, 0.54117647],
           [0.62352941, 0.21176471, 0.12941176],
           [0.31372549, 0.        , 0.58039216],
           [0.74117647, 0.56470588, 0.50196078],
           [0.        , 0.42745098, 0.49411765],
           [0.34509804, 0.8745098 , 0.37647059],
           [0.27843137, 0.31372549, 0.40392157],
           [0.00392157, 0.36470588, 0.62352941],
           [0.38823529, 0.18823529, 0.23529412],
           [0.00784314, 0.80784314, 0.58039216],
           [0.54509804, 0.3254902 , 0.14509804],
           [0.67058824, 0.        , 1.        ],
           [0.55294118, 0.16470588, 0.52941176],
           [0.33333333, 0.3254902 , 0.58039216],
           [0.58823529, 1.        , 0.        ],
           [0.        , 0.59607843, 0.48235294],
           [1.        , 0.54117647, 0.79607843],
           [0.87058824, 0.27058824, 0.78431373],
           [0.41960784, 0.42745098, 0.90196078],
           [0.11764706, 0.        , 0.26666667],
           [0.67843137, 0.29803922, 0.54117647],
           [1.        , 0.5254902 , 0.63137255],
           [0.        , 0.1372549 , 0.23529412],
           [0.54117647, 0.80392157, 0.        ],
           [0.43529412, 0.79215686, 0.61568627],
           [0.88235294, 0.29411765, 0.99215686],
           [1.        , 0.69019608, 0.30196078],
           [0.89803922, 0.90980392, 0.22352941],
           [0.44705882, 0.0627451 , 1.        ],
           [0.43529412, 0.32156863, 0.39607843],
           [0.5254902 , 0.5372549 , 0.18823529],
           [0.38823529, 0.14901961, 0.31372549],
           [0.41176471, 0.14901961, 0.1254902 ],
           [0.78431373, 0.43137255, 0.        ],
           [0.81960784, 0.64313725, 1.        ],
           [0.77647059, 0.82352941, 0.3372549 ],
           [0.30980392, 0.40392157, 0.30196078],
           [0.68235294, 0.64705882, 0.65098039],
           [0.66666667, 0.17647059, 0.39607843],
           [0.78039216, 0.31764706, 0.68627451],
           [1.        , 0.34901961, 0.6745098 ],
           [0.57254902, 0.4       , 0.30588235],
           [0.4       , 0.5254902 , 0.72156863],
           [0.43529412, 0.59607843, 1.        ],
           [0.36078431, 1.        , 0.62352941],
           [0.6745098 , 0.5372549 , 0.69803922],
           [0.82352941, 0.13333333, 0.38431373],
           [0.78039216, 0.81176471, 0.57647059],
           [1.        , 0.7254902 , 0.11764706],
           [0.98039216, 0.58039216, 0.55294118],
           [0.19215686, 0.13333333, 0.30588235],
           [0.99607843, 0.31764706, 0.38039216],
           [0.99607843, 0.55294118, 0.39215686],
           [0.26666667, 0.21176471, 0.09019608],
           [0.78823529, 0.63529412, 0.32941176],
           [0.78039216, 0.90980392, 0.94117647],
           [0.26666667, 0.59607843, 0.        ],
           [0.57647059, 0.6745098 , 0.22745098],
           [0.08627451, 0.29411765, 0.10980392],
           [0.03137255, 0.32941176, 0.4745098 ],
           [0.45490196, 0.17647059, 0.        ],
           [0.40784314, 0.23529412, 1.        ],
           [0.25098039, 0.16078431, 0.14901961],
           [0.64313725, 0.44313725, 0.84313725],
           [0.81176471, 0.        , 0.60784314],
           [0.4627451 , 0.00392157, 0.1372549 ],
           [0.3254902 , 0.        , 0.34509804],
           [0.        , 0.32156863, 0.90980392],
           [0.16862745, 0.36078431, 0.34117647],
           [0.62745098, 0.85098039, 0.57254902],
           [0.69019608, 0.10196078, 0.89803922],
           [0.11372549, 0.01176471, 0.14117647],
           [0.47843137, 0.22745098, 0.62352941],
           [0.83921569, 0.81960784, 0.81176471],
           [0.62745098, 0.39215686, 0.41176471],
           [0.41568627, 0.61568627, 0.62745098],
           [0.6       , 0.85882353, 0.44313725],
           [0.75294118, 0.21960784, 0.81176471],
           [0.49019608, 1.        , 0.34901961],
           [0.58431373, 0.        , 0.13333333],
           [0.83529412, 0.63529412, 0.8745098 ],
           [0.08627451, 0.51372549, 0.8       ],
           [0.65098039, 0.97647059, 0.27058824],
           [0.42745098, 0.41176471, 0.38039216],
           [0.3372549 , 0.7372549 , 0.30588235],
           [1.        , 0.42745098, 0.31764706],
           [1.        , 0.01176471, 0.97254902],
           [1.        , 0.        , 0.28627451],
           [0.79215686, 0.        , 0.1372549 ],
           [0.2627451 , 0.42745098, 0.07058824],
           [0.91764706, 0.66666667, 0.67843137],
           [0.74901961, 0.64705882, 0.        ],
           [0.14901961, 0.17254902, 0.2       ],
           [0.33333333, 0.7254902 , 0.00784314],
           [0.4745098 , 0.71372549, 0.61960784],
           [0.99607843, 0.9254902 , 0.83137255],
           [0.54509804, 0.64705882, 0.34901961],
           [0.55294118, 0.99607843, 0.75686275],
           [0.        , 0.23529412, 0.16862745],
           [0.24705882, 0.06666667, 0.15686275],
           [1.        , 0.86666667, 0.96470588],
           [0.06666667, 0.10196078, 0.57254902],
           [0.60392157, 0.25882353, 0.32941176],
           [0.58431373, 0.61568627, 0.93333333],
           [0.49411765, 0.50980392, 0.28235294],
           [0.22745098, 0.02352941, 0.39607843],
           [0.74117647, 0.45882353, 0.39607843]]
