from __future__ import division

import os
import numpy as np
import pandas as pd

from helper_lib import filelib


class Metadata(pd.Series):
    """
    Class for a cell metadata.
    """

    def __init__(self, filename=None):
        """
        Initializes the Metadata class by loading from file or generating inputs from a give resolution value.
        
        Parameters
        ----------
        filename : str, optional
            Path used to load the metadata from file.
            If None, no metadata will be loaded.
            Default is None.
        """
        super(Metadata, self).__init__()

        if filename is not None:
            self['Name'] = filename
            self.read_from_csv(filename)

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
        self.to_csv(outputfile, sep='\t', header=False)

    def read_from_csv(self, filename):
        """
        Reads cell metadata from a csv file.
        
        Parameters
        ----------
        filename : str
            The path used to read the cell metadata.
        """
        if os.path.exists(filename):
            f = open(filename)
            st = f.readlines()
            f.close()
            if len(st) > 0:
                data = pd.read_csv(filename, sep='\t', index_col=0, header=-1).transpose()
                if len(data.columns == 1):
                    data = data.iloc[0]
                else:
                    data = data.iloc[0].T.squeeze()

                for c in data.index:
                    try:
                        self[c] = float(data[c])
                    except ValueError:
                        self[c] = data[c]
                if 'Voxel size x' in data.index and 'Voxel size y' in data.index and 'Voxel size z' in data.index:
                    self['Voxel size'] = str(np.float_([self['Voxel size z'],
                                                        self['Voxel size y'], self['Voxel size x']]))
                    self['Voxel size arr'] = np.array([self['Voxel size z'], self['Voxel size y'],
                                                       self['Voxel size x']])

    def set_voxel_size(self, voxel_size):
        """
        Converts voxel size

        Parameters
        ----------
        voxel_size : scalar or sequence of scalars
            Voxel size in z, y and x used to generate the cell image.
            If one number is provided, the same value will be used for all dimensions.
        """
        voxel_size = np.array([voxel_size]).flatten()
        if len(voxel_size) == 1:
            voxel_size = np.array([voxel_size[0], voxel_size[0], voxel_size[0]])
        elif len(voxel_size) == 2:
            voxel_size = np.array([voxel_size[0], voxel_size[1], voxel_size[1]])
        elif len(voxel_size) == 3:
            voxel_size = voxel_size
        else:
            raise ValueError('voxel size must be a number of a sequence of length 2 or 3!')
        self['Voxel size'] = str(np.float_(voxel_size))

        dimension_xyz = ['z', 'y', 'x']

        for i, n in enumerate(voxel_size):
            self['Voxel size ' + dimension_xyz[i]] = float(n)

        self['Voxel size arr'] = np.array([self['Voxel size z'], self['Voxel size y'], self['Voxel size x']])



