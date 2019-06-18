from __future__ import division

import os
import pandas as pd
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage import io
import pylab as plt
from skimage.exposure import rescale_intensity
import warnings

from DeconvTest.classes.metadata import Metadata
from DeconvTest.modules import noise
from DeconvTest.modules import quantification
from helper_lib import filelib


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
        self.metadata = Metadata()
        self.metadata['Convolved'] = False
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
            if os.path.exists(filename[:-4] + '.csv'):
                self.metadata.read_from_csv(filename[:-4] + '.csv')
            else:
                warnings.warn("Metadata file is not found!", Warning)
        else:
            raise ValueError('File does not exist!')

    def normalize(self):
        """
        Normalized the current image between 0 and 255.
        """
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
                io.imsave(outputfile, self.image.astype(np.uint32))
        self.metadata.to_csv(outputfile[:-4] + '.csv', sep='\t', header=False)
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

    def save_projection(self, outputfile, axis=1):
        """
        Saves the maximum intensity projection of the stack.

        Parameters
        ----------
        outputfile : str
            The path used to save the maximum intensity projection.
        axis : int, optional
            Axis along which the projection should be made.
            Default is 1 (xz).
        """
        filelib.make_folders([os.path.dirname(outputfile)])
        maxproj = np.max(self.image, axis=axis)
        io.imsave(outputfile, maxproj.astype(np.uint8))

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
        self.metadata['Convolved'] = True
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

        if 'Voxel size x' in self.metadata.index and 'Voxel size y' in self.metadata.index \
                and 'Voxel size z' in self.metadata.index:
            new_voxel_size = np.array([self.metadata['Voxel size z'], self.metadata['Voxel size y'],
                                                     self.metadata['Voxel size x']]) / kwargs['zoom']
            self.metadata['Voxel size'] = str(new_voxel_size)
            self.metadata['Voxel size z'], self.metadata['Voxel size y'], self.metadata['Voxel size x'] = new_voxel_size

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
        snr : float or sequence of floats, optional
            Target signal-to-noise ratio (SNR) for each noise type.
            Must be the same shape as the `kind` argument.
            If None, no noise is added.
            Default is None

        Returns
        -------
        ndarray
            Output noisy image of the same shape as the current image.
        """
        if kind is not None:
            if type(kind) is str:
                kind = [kind]
            snr = np.array([snr]).flatten()
            if len(kind) != len(snr):
                if len(snr) == 1:
                    snr = np.ones(len(kind)) * snr
                else:
                    raise TypeError("The length of the array for SNR must be the same as for the noise type!")
            for i, k in enumerate(kind):
                if 'add_' + k + '_noise' in dir(noise) and k in noise.valid_noise_types:
                    self.image = getattr(noise, 'add_' + k + '_noise')(img=self.image, snr=snr[i])
                    if i == 0:
                        ind = ''
                    else:
                        ind = ' ' + str(i + 1)
                    self.metadata['SNR' + ind] = snr[i]
                    self.metadata['noise type' + ind] = k
                else:
                    raise AttributeError(k + ' is not a valid noise type!')

        return self.image

    def compute_accuracy_measures(self, gt):
        """
        Computes the root mean square error (RMSE) and it normalized versions

        Parameters
        ----------
        gt : Image or Cell
            Ground truth image.

        Returns
        -------
        pandas.DataFrame()
            Data frame containing the values for the computed accuracy measures.
        """
        data = quantification.compute_accuracy_measures(self.image, gt.image)
        return data
