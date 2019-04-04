from __future__ import division

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from image import Image


class PSF(Image):
    """
    Class for a 3D image of a point spread function (PSF).
    """

    def __init__(self, sigma=None, elongation=None, filename=None):
        """
        Initializes the PSF image by reading from file or generating from given sigma and elongation
        
        Parameters
        ----------
        sigma : float, optional
            Standard deviation in xy in pixels of the Gaussian function that is used to approximate the PSF.
            If None, no image will be generated.
            Default is None.
        elongation : float, optional
            Ratio between the Gaussian standard deviations in z and xy.
            If None, no image will be generated.
            Default is None.
        filename : str, optional
            Path used to load the PSF image.
            If None or non-existent, no image will be loaded.
            Default is None.
        """
        super(PSF, self).__init__(filename=filename)
        if sigma is not None and elongation is not None:
            self.generate(sigma, elongation)

    def __repr__(self):
        return "Point spread function image"

    def generate(self, sigma, elongation):
        """
        Generates a Point Spread Function (PSF) with a given standard deviation and elongation (aspect ratio).

        Parameters
        ----------
        sigma : float
            Standard deviation in xy in pixels of the Gaussian function that is used to approximate the PSF.
        elongation : float
            Ratio between the Gaussian standard deviations in z and xy.

        Returns
        -------
        ndarray
            Output 3D image of the PSF.
        """

        # set image dimensions 8 times the standard deviation of the Gaussian
        scale = 8
        zsize = int(round((sigma + 1) * elongation)) * scale + 1
        xsize = int(round(sigma + 1)) * scale + 1

        x = np.zeros([zsize, xsize, xsize])  # create an empty array
        x[int(zsize / 2), int(xsize / 2), int(xsize / 2)] = 255.  # create a peak in the center of the image
        x = ndimage.gaussian_filter(x, [sigma * elongation, sigma, sigma])  # smooth the peak with a Gaussian
        x = x / np.max(x)

        self.image = x
        return x

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
            coeff, var_matrix = curve_fit(self.__gauss, x, profile, p0=p0, bounds=bounds)  # fit the Gauss curve
            sigmas.append(coeff[-1])

        return np.array(sigmas)

    def __gauss(self, x, *p):
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










