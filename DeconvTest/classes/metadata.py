from __future__ import division

import os
import re
import numpy as np
import pandas as pd
import warnings

from helper_lib import filelib


class Metadata(pd.Series):
    """
    Class for a cell metadata.
    """

    def __init__(self, resolution=None, filename=None, string=None):
        """
        Initializes the Metadata class by loading from file or generating inputs from a give resolution value.
        
        Parameters
        ----------
        resolution : scalar or sequence of scalars
            Voxel size in z, y and x used to generate the cell image.
        filename : str, optional
            Path used to load the metadata from file.
            If None, no metadata will be loaded.
            Default is None.
        string : str, optional
            String to extract metadata from.
            Default is None.
        """
        super(Metadata, self).__init__()
        if resolution is not None:
            stat = self.__resolution_to_voxel_size(resolution, pd.Series())
            for c in stat.index:
                self[c] = stat[c]
        else:
            self['resolution'] = None

            for i in ['x', 'y', 'z']:
                self['Voxel size ' + i] = None

        if filename is not None:
            self['Name'] = filename
            self.read_from_csv(filename)
        if string is not None:
            self.parse_from_string(string)

    def __repr__(self):
        return "Metadata of an image"

    def save(self, outputfile):
        """
        Saves the cell metadata to a csv file

        Parameters
        ----------
        outputfile: str
            The path used to save the cell metadata.
        """
        filelib.make_folders([os.path.dirname(outputfile)])
        self.to_csv(outputfile, sep='\t')

    def read_from_csv(self, filename):
        """
        Reads cell metadata from a csv file.
        
        Parameters
        ----------
        filename : str
            The path used to read the cell metadata.
        """
        if os.path.exists(filename):
            data = pd.read_csv(filename, sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze()
            super(Metadata, self).__init__()

            for c in data.index:
                try:
                    self[c] = float(data[c])
                except ValueError:
                    self[c] = data[c]
            self['resolution'] = [self['Voxel size z'], self['Voxel size y'], self['Voxel size x']]

    def parse_from_string(self, string):

        stat = pd.Series({'Name': string,
                          'CellID': None,
                          'Kind': None,
                          'PSF sigma X': None,
                          'PSF elongation': None,
                          'PSF sigma Z': None,
                          'Poisson_SNR': 'None',
                          'Gaussian_SNR': 'None',
                          'Lambda': None,
                          'Iterations': None,
                          'Perform anti-ringing': None,
                          'Normalize PSF': None,
                          'Detect divergence': None,
                          'Low pass filter, pixels': None,
                          'Wiener filter gamma': None,
                          'Terminate if mean delta <': None})  # initialize pandas Series with None values

        p = re.compile('\d*\.*\d+')  # create a template for extracting nonnegative float numbers
        nums = p.findall(string)
        if len(nums) > 0:
            num = float(nums[-1])
            if not num.is_integer():
                raise ValueError("ID of a cell must be integer!")
            stat['CellID'] = int(num)  # extract Cell ID
            stat['Kind'] = 'Input'

        else:
            stat['CellID'] = None
            warnings.warn("File name does not contain cell ID!", Warning)

        # extract PSF parameters
        parts = string.split('sigma')  # find the keyword for PSF sigma
        if len(parts) > 1:  # if found, extract the first number after the keyword, and set the 'kind' as convolved
            stat['PSF sigma X'] = float(p.findall(parts[1].split('/')[0])[0])
            stat['Kind'] = 'Convolved'

        parts = string.split('elongation')  # find the keyword for PSF elongation
        if len(parts) > 1:  # if found, extract the first number after the keyword
            stat['PSF elongation'] = float(p.findall(parts[1].split('/')[0])[0])

        if not (stat['PSF sigma X'] is None or stat['PSF elongation'] is None):  # compute PSF sigma in z
            stat['PSF sigma Z'] = stat['PSF sigma X'] * stat['PSF elongation']

        resolution = None
        # Extract voxel size (resolution)
        parts = string.split('voxel')  # find the keyword for voxel
        if len(parts) == 1:
            parts = string.split('pixel')  # or find the keyword for pixel

        if len(parts) > 1:  # if found, extract all the numbers after the keyword and before '/'-symbol
            nums = p.findall(parts[1].split('/')[0].split('gaussian')[0])

            if len(nums) > 0:  # if found, convert the extracted numbers to float
                resolution = np.float_(nums)

        if resolution is not None:  # if resolution has been extracted from image name or provided as a keyword argument
            stat = self.__resolution_to_voxel_size(resolution, stat)

        # Extract SNR information
        if len(string.split('gaussian_snr')) > 1:  # find the keyword for SNR
            stat['Gaussian_SNR'] = string.split('gaussian_snr=')[1].split('_')[0].split('/')[
                0]  # extract the first number after the keyword

        if len(string.split('poisson_snr')) > 1:  # find the keyword for SNR
            stat['Poisson_SNR'] = string.split('poisson_snr=')[1].split('_')[0].split('/')[
                0]  # extract the first number after the keyword

        # Extract the ImageJ settings
        p1 = re.compile('[-+]?\d*e[-+]?\d+')  # create a template for extracting float numbers with e
        parts = string.split('lambda=')  # find the keyword for lambda
        if len(parts) > 1:  # if found, extract the first number after the keyword
            if len(p1.findall(parts[1])) > 0:
                stat['Lambda'] = float(p1.findall(parts[1])[0])
            else:
                stat['Lambda'] = float(p.findall(parts[1])[0])

        parts = string.split('iterations=')  # find the keyword for iterations
        if len(parts) > 1:  # if found, extract the first number after the keyword
            stat['Iterations'] = int(float(p.findall(parts[1])[0]))

        parts = string.split('perform=')
        if len(parts) > 1:
            stat['Perform anti-ringing'] = parts[1].split('_')[0]

        parts = string.split('normalize=')
        if len(parts) > 1:
            stat['Normalize PSF'] = parts[1].split('_')[0]

        parts = string.split('detect=')
        if len(parts) > 1:
            stat['Detect divergence'] = parts[1].split('_')[0]

        parts = string.split('low=')
        if len(parts) > 1:
            stat['Low pass filter, pixels'] = int(float(p.findall(parts[1])[0]))

        parts = string.split('wiener=')
        if len(parts) > 1:
            stat['Wiener filter gamma'] = float(p.findall(parts[1])[0])

        parts = string.split('terminate=')
        if len(parts) > 1:
            stat['Terminate if mean delta <'] = float(p.findall(parts[1])[0])

        for c in ['Voxel size x', 'Voxel size y', 'Voxel size z', 'resolution']:
            if c in stat.index:
                self[c] = stat[c]
        for c in stat.index:
            if c not in self.index:
                self[c] = stat[c]

    def __resolution_to_voxel_size(self, resolution, stat):

        resolution = np.array([resolution]).flatten()
        if len(resolution) == 1:
            resolution = np.array([resolution[0], resolution[0], resolution[0]])
        elif len(resolution) == 2:
            resolution = np.array([resolution[0], resolution[1], resolution[1]])
        elif len(resolution) == 3:
            resolution = resolution
        else:
            raise ValueError('resolution must be a number of a sequence of length 2 or 3!')
        stat['resolution'] = resolution

        dimension_xyz = ['z', 'y', 'x']

        for i, n in enumerate(resolution):
            stat['Voxel size ' + dimension_xyz[i]] = float(n)

        return stat

