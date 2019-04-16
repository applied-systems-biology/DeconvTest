"""
This module contains functions for generating synthetic noise
"""

import numpy as np

valid_noise_types = ['gaussian', 'poisson']


def add_poisson_noise(img, snr=None):
    """
    Adds random Poisson noise to the current image.

    Parameters
    ----------
    img : ndarray
        Input image
    snr : float, optional
        Target signal-to-noise ratio (SNR) after adding the noise.
        If None, no noise is added.
        Default is None

    Returns
    -------
    ndarray
        Output noisy image of the same shape as the current image.
    """

    if img is None:
        raise ValueError('self.image is None! The image has to be initialized!')

    if snr is not None:
        imgmax = snr ** 2
        ratio = imgmax / img.max()
        img = img * 1. * ratio
        img = np.random.poisson(img)
        img = img / ratio
    return img


def add_gaussian_noise(img, snr=None):
    """
    Adds random Gaussian noise to the current image.

    Parameters
    ----------
    img : ndarray
        Input image
    snr : float, optional
        Target signal-to-noise ratio (SNR) after adding the noise.
        If None, no noise is added.
        Default is None

    Returns
    -------
    ndarray
        Output noisy image of the same shape as the current image.
    """

    if img is None:
        raise ValueError('self.image is None! The image has to be initialized!')

    if snr is not None:
        sig = img.max() * 1. / (10 ** (snr / 20.))
        noise = np.random.normal(0, sig, img.shape)
        img = img + noise
        img[np.where(img < 0)] = 0
    return img
