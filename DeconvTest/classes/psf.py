from __future__ import division

import numpy as np
from scipy.optimize import curve_fit

from DeconvTest.classes.image import Image
from DeconvTest.modules import psfs


class PSF(Image):
    """
    Class for a 3D image of a point spread function (PSF).
    """

    def __init__(self, filename=None, **kwargs):
        """
        Initializes the PSF image by reading from file or generating from given sigma and elongation
        
        Parameters
        ----------
        filename : str, optional
            Path used to load the PSF image.
            If None or non-existent, no image will be loaded.
            Default is None.
        kwargs : key, value pairings
            Keyword arguments passed to corresponding methods to generate synthetic PSFs.

        """
        super(PSF, self).__init__(filename=filename)
        if self.image is None:
            self.generate(**kwargs)

    def __repr__(self):
        return "Point spread function image"

    def generate(self, kind='gaussian', **kwargs):
        """
        Generates a Point Spread Function (PSF) with a given standard deviation and elongation (aspect ratio).

        Parameters
        ----------
        kind : string, optional
            Name of the shape of the PSF from set of
            {gaussian}.
            Default is 'gaussian'
        kwargs : key, value pairings
            Keyword arguments passed to corresponding methods to generate synthetic PSFs.

        Returns
        -------
        ndarray
            Output 3D image of the PSF.
        """

        # set image dimensions 8 times the standard deviation of the Gaussian

        if kind in dir(psfs) and kind in psfs.valid_shapes:
            self.image = getattr(psfs, kind)(**kwargs)
        else:
            raise AttributeError(kind + ' is not a valid object shape!')

        return self.image

    def measure_psf_sigma(self):
        """
        Measures the standard deviation of the current PSF image.

        Returns
        -------
        array of floats
            Measured standard deviation for all 3 axes in pixels.
        """
        profile0 = self.image.max(1).max(1)  # profile of maxumum values along z axis
        profile1 = self.image.max(2).max(0)  # profile of maxumum values along y axis
        profile2 = self.image.max(0).max(0)  # profile of maxumum values along x axis

        sigmas = []
        for profile in [profile0, profile1, profile2]:
            x = np.arange(len(profile))  # array of indices: independent variable for fitting
            p0 = [profile.max(), profile.argmax(), 1.]  # initial values for amplitude, center and standard deviation
            r = profile.max() - profile.min()  # range of values in the profile
            bounds = ([p0[0] - r / 4, p0[1] - len(profile) / 4, 0],
                      [p0[0] + r / 4, p0[1] + len(profile) / 4,
                       1000])  # set the lower and upper boundaries for parameters
            coeff, var_matrix = curve_fit(self.__gauss_fit, x, profile, p0=p0, bounds=bounds)  # fit the Gauss curve
            sigmas.append(coeff[-1])

        return np.array(sigmas)

    def __gauss_fit(self, x, *p):
        """
        Returns a Gaussian function of the input `x` with parameters `p`
    
        Parameters
        ----------
        x : scalar or sequence of scalars
            The input (independent variable) of the Gaussian 
        a: float
            Amplitude of the Gaussian
        mu: float
            The center (mean value) of the Gaussian
        sigma: float
            The standard deviation of the Gaussian
    
        Returns
        -------
        scalar or sequence of scalars
            Gaussian function of `x` with the same shape as `x`
        """
        a, mu, sigma = p
        return a * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))










