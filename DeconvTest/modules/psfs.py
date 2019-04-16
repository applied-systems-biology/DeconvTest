"""
This module contains functions for generating synthetic PSF images
"""

from __future__ import division
import numpy as np
from scipy import ndimage

valid_shapes = ['gaussian']


def gaussian(sigma=None, aspect_ratio=None, **kwargs_to_ignore):
    """
    Generates a Point Spread Function (PSF) with a Gaussian shape and given standard deviation and aspect ratio.

    Parameters
    ----------
    sigma : float
        Standard deviation in xy in pixels of the Gaussian function that is used to approximate the PSF.
        If None, no PSF will be generated.
        Default is None.
    aspect_ratio : float
        Ratio between the Gaussian standard deviations in z and xy.
        If None, no PSF will be generated.
        Default is None.

    Returns
    -------
    ndarray
        Output 3D image of the PSF.
    """

    # set image dimensions 8 times the standard deviation of the Gaussian
    if sigma is not None and aspect_ratio is not None:
        scale = 8
        zsize = int(round((sigma + 1) * aspect_ratio)) * scale + 1
        xsize = int(round(sigma + 1)) * scale + 1

        x = np.zeros([zsize, xsize, xsize])  # create an empty array
        x[int(zsize / 2), int(xsize / 2), int(xsize / 2)] = 255.  # create a peak in the center of the image
        x = ndimage.gaussian_filter(x, [sigma * aspect_ratio, sigma, sigma])  # smooth the peak with a Gaussian
        x = x / np.max(x)

    else:
        x = None
    return x










