from __future__ import division

import os
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage import io
import pylab as plt
import warnings

from metadata import Metadata
from helper_lib import filelib
from skimage.exposure import rescale_intensity


class Image(object):
    """
    Class for a 3D image.

    """

    def __init__(self, filename=None, image=None):
        """
        Initializes the Image class

        Parameters
        ----------
        filename: str, optional
            The path used to load the image.
            If None, an empty class is created.
            If `image` argument is provided, the image will be initialized from the `image` argument.
            Default is None.
        image : ndarray, optional
            3D array to initiate the image.
            If None, an empty class is created.
            Default is None.
        """
        self.image = image
        self.filename = filename
        self.metadata = None
        if self.image is None and filename is not None and os.path.exists(filename):  # read the image from file
            self.from_file(filename)

    def __repr__(self):
        return "3D image"

    def from_file(self, filename):
        """
        Reads the image from file.

        Parameters
        ----------
        filename: str
            Path used to load the image and extract metadata.
        """
        if os.path.exists(filename):  # read the image from file
            self.image = io.imread(filename)
            self.filename = filename
            self.metadata = Metadata()
            if os.path.exists(filename[:-4] + '.csv'):
                self.metadata.read_from_csv(filename[:-4] + '.csv')
        else:
            raise ValueError('File does not exist!')

    def normalize(self):
        self.image = rescale_intensity(self.image, out_range=(0, 255))

    def save(self, outputfile, normalize_output=False):
        """
        Saves the image to a given file.

        Parameters
        ----------
        outputfile: str
            Path to save the image.
        normalize_output : bool, optional
            If True, the output image will be normalized to 8 bit before saving.
            Default is False.
        """
        filelib.make_folders([os.path.dirname(outputfile)])
        if not outputfile.endswith('.tif'):
            outputfile += '.tif'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if normalize_output:
                io.imsave(outputfile, rescale_intensity(self.image, out_range=(0, 255)).astype(np.uint8))
            else:
                io.imsave(outputfile, self.image.astype(np.uint8))
        self.filename = outputfile

    def show_2d_projections(self):
        """
        Shows maximum projection of a cell in xy, yz, and xz.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, axs = plt.subplots(1, 3)
            for i in range(3):
                plt.sca(axs[i])
                io.imshow(self.image.max(i), cmap='viridis')

    def convolve(self, psf):
        """
        Convolves the current image with a given point spread function (PSF).
        
        Parameters
        ----------
        psf : PSF
            Instance of the class PSF containing the image of the point spread function.
            
        Returns
        -------
        ndarray
            Convolved image.
        """
        if self.image is None or psf.image is None:
            raise ValueError('Both images to convolve have to be initialized!')
        self.image = fftconvolve(self.image, psf.image, mode='full')
        self.image = rescale_intensity(self.image, out_range=(0, 255))
        return self.image

    def resize(self, **kwargs):
        """
        Resizes the image according to the given zoom and interpolation order, 
         pads the output with zeros if one of the axes dimensions <=3
        
        Parameters
        ----------
        kwargs : key, value pairings
            Keyword arguments passed to `ndimage.interpolate.zoom` function.
            
        Returns
        -------
        ndarray
            Resized image.
        """

        if self.image is None:
            raise ValueError('self.image is None! The image has to be initialized!')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.image = ndimage.interpolation.zoom(self.image * 1., **kwargs)

        # if size <= 3, pad with zeros

        if np.min(self.image.shape) < 5:
            self.image = np.pad(self.image, pad_width=3, mode='constant', constant_values=0)

        if self.image.max() > 0:
            self.image = rescale_intensity(self.image, out_range=(0, 255))
        return self.image

    def add_noise(self, kind=None, snr=None):
        """
        Adds random Poisson noise to the current image.

        Parameters
        ----------
        kind : string, sequence of strings or None, optional
            Name of the method to generate nose from set of {gaussian, poisson}.
            If a sequence is provided, several noise types will be added.
            If None, no noise will be added.
            Default is None.
        snr : float, optional
            Target signal-to-noise ratio (SNR) after adding the noise.
            If None, no noise is added.
            Default is None

        Returns
        -------
        ndarray
            Output noisy image of the same shape as the current image.
        """
        valid_noise_types = ['gaussian', 'poisson']
        if kind is not None:
            if type(kind) is str:
                kind = [kind]
            for k in kind:
                if 'add_' + k + '_noise' in dir(self) and k in valid_noise_types:
                    self.image = getattr(self, 'add_' + k + '_noise')(snr=snr)
                else:
                    raise AttributeError(k + ' is not a valid noise type!')

        return self.image

    def add_poisson_noise(self, snr=None):
        """
        Adds random Poisson noise to the current image.

        Parameters
        ----------
        snr : float, optional
            Target signal-to-noise ratio (SNR) after adding the noise.
            If None, no noise is added.
            Default is None

        Returns
        -------
        ndarray
            Output noisy image of the same shape as the current image.
        """

        if self.image is None:
            raise ValueError('self.image is None! The image has to be initialized!')

        if snr is not None:
            imgmax = snr ** 2
            ratio = imgmax / self.image.max()
            self.image = self.image * 1. * ratio
            self.image = np.random.poisson(self.image)
            self.image = self.image / ratio
        return self.image

    def add_gaussian_noise(self, snr=None):
        """
        Adds random Gaussian noise to the current image.

        Parameters
        ----------
        snr : float, optional
            Target signal-to-noise ratio (SNR) after adding the noise.
            If None, no noise is added.
            Default is None

        Returns
        -------
        ndarray
            Output noisy image of the same shape as the current image.
        """

        if self.image is None:
            raise ValueError('self.image is None! The image has to be initialized!')

        if snr is not None:
            sig = self.image.max() * 1. / (10 ** (snr / 20.))
            noise = np.random.normal(0, sig, self.image.shape)
            self.image = self.image + noise
            self.image[np.where(self.image < 0)] = 0
        return self.image
