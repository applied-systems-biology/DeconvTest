from __future__ import division

import os
import numpy as np
import re
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
            if 'Voxel size x' in data.index and 'Voxel size y' in data.index and 'Voxel size z' in data.index:
                self['Voxel size'] = [self['Voxel size z'], self['Voxel size y'], self['Voxel size x']]
            self.__convert_voxel_size()

    def __convert_voxel_size(self):

        p = re.compile('\d*\.*\d+')
        if 'Voxel size' in self.index and type(self['Voxel size']) is str:
            self['Voxel size'] = np.float_(p.findall(self['Voxel size']))

