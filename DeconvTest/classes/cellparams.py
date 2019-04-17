from __future__ import division

import os
import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from DeconvTest.modules import input_objects
from helper_lib import filelib


class CellParams(pd.DataFrame):
    """
    Class for generation and storing parameters for a set of cells.
    """

    def __init__(self, **kwargs):
        """
        Initializes the class for CellParams and generate random cell parameters with given properties.

        Keyword arguments:
        -----------    
        number_of_cells: int
            Number of cells to generate.
        size_mean_and_std: tuple
            Mean value and standard deviation for the cell size in micrometers.
            The cell size is drawn randomly from a Gaussian distribution with the given mean and standard deviation.
        equal_dimensions: bool
            If True, generates parameters for a sphere.
            If False, generate parameters for an ellipsoid with sizes for all three axes chosen independently.
        coordinates : bool
            If True, relative cell coordinates will be generated in the range from 0 to 1.
        spikiness_range : tuple
            Range for the fraction of cell surface area covered by spikes.
        spike_size_range : tuple
            Range for the standard deviation for the spike amplitude relative to the cell radius.
        spike_smoothness_range : tuple
            Range for the width of the Gaussian filter that is used to smooth the spikes.
        
        """
        super(CellParams, self).__init__()
        self.generate_parameters(**kwargs)

    def generate_parameters(self, kind='ellipsoid', number_of_stacks=None, number_of_cells=1, coordinates=True, **kwargs):
        """
        Generates random cells sizes and rotation angles.

        Parameters:
        -----------
        kind : string, optional
            Name of the shape of the ground truth object from set of
            {ellipoid, spiky_cell}.
            Default is 'ellipsoid'
        number_of_stacks : int, optional
            Number of stacks to generate.
            If None, parameters for single cells will be generated
            Default is None.
        number_of_cells: int, optional
            Number of cells to generate.
            Default is 1.
        coordinates : bool, optional
            If True, relative cell coordinates will be generated.
            Default is True.
        kwargs : key, value pairings
            Keyword arguments passed to corresponding methods to generate cell parameters.
        """

        if 'parameters_' + kind in dir(input_objects) and kind in input_objects.valid_shapes:
            data = pd.DataFrame()
            number_of_cells = np.array([number_of_cells]).flatten()
            if number_of_stacks is None:
                if not len(number_of_cells) == 1:
                    raise ValueError("Number of cells must be integer!")
            else:
                if len(number_of_cells) == 1:
                    pass
                elif len(number_of_cells) == 2:
                    number_of_cells = np.random.randint(number_of_cells[0], number_of_cells[1], number_of_stacks)
                else:
                    raise ValueError("Number of cells must be integer or tuple of two integers!")

            iterations = number_of_stacks
            if iterations is None:
                iterations = 1
            for i_iter in range(iterations):
                cells = pd.DataFrame()
                for i_num in range(int(number_of_cells)):
                    cell = getattr(input_objects, 'parameters_' + kind)(**kwargs)
                    if coordinates:
                        cell.loc[:, 'z'], cell.loc[:, 'y'], cell.loc[:, 'x'] = np.random.uniform(0, 1, 3)
                    cell.loc[:, 'kind'] = kind

                    cells = pd.concat([cells, cell], ignore_index=True)

                if number_of_stacks is not None:
                    cells.loc[:, 'stack'] = i_iter
                data = pd.concat([data, cells], ignore_index=True)
            for c in data.columns:
                self.loc[:, c] = data[c]

        else:
            raise AttributeError(kind + ' is not a valid object shape!')

    def save(self, outputfile):
        """
        Saves the cell parameters to a csv file

        Parameters
        ----------
        outputfile: str
            The path used to save the cell parameters.
        """
        filelib.make_folders([os.path.dirname(outputfile)])
        self.to_csv(outputfile, sep='\t')

    def read_from_csv(self, filename):
        """
        Reads cell parameters from a csv file.
        
        Parameters
        ----------
        filename : str
            The path used to read the cell parameters.
        """
        if os.path.exists(filename):
            data = pd.read_csv(filename, sep='\t', index_col=0)
            super(CellParams, self).__init__()

            for c in data.columns:
                self[c] = data[c]

    def plot_size_distribution(self):
        """
        Plots pairwise distributions of the cell sizes along x, y and z axes.

        Return
        ------
        seaborn.pairplot
            Plot with the pairwise distributions, which can be displayed or saved to a file.
        """

        cells = self
        plt.close()
        pl = sns.pairplot(cells, vars=['size_x', 'size_y', 'size_z'])

        return pl.fig

    def plot_angle_distribution(self):
        """
        Plots pairwise distributions of azimuthal and polar rotation angles.

        Return
        ------
        seaborn.pairplot
            Plot with the pairwise distributions, which can be displayed or saved to a file.
        """

        cells = self
        plt.close()
        pl = sns.pairplot(cells, vars=['phi', 'theta'])

        return pl.fig

