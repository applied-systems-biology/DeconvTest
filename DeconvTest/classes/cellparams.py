from __future__ import division

import os
import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helper_lib import filelib


class CellParams(pd.DataFrame):
    """
    Class for generation and storing parameters for a set of cells.
    """

    def __init__(self, number_of_cells=1, size_mean_and_std=(10, 2), equal_dimensions=False, coordinates=True,
                 spikiness_range=(0, 0), spike_size_range=(0, 0), spike_smoothness_range=(0.05, 0.1)):
        """
        Initializes the class for CellParams and generate random cell parameters with given properties.

        Parameters:
        -----------    
        number_of_cells: int, optional
            Number of cells to generate.
            Default is 1.
        size_mean_and_std: tuple, optional
            Mean value and standard deviation for the cell size in micrometers.
            The cell size is drawn randomly from a Gaussian distribution with given mean and standard deviation.
            Default is (10, 2).
        equal_dimensions: bool, optional
            If True, generates parameters for a sphere.
            If False, generate parameters for an ellipsoid with sizes for all three axes chosen independently.
            Default is True
        coordinates : bool, optional
            If True, relative cell coordinates will be generated.
            Default is True.
        spikiness_range : tuple, optional
            Range for the fraction of cell surface area covered by spikes.
            Default is (0, 0).
        spike_size_range : tuple, optional
            Range for the standard deviation for the spike amplitude relative to the cell radius.
            Default is (0, 0).
        spike_smoothness_range : tuple, optional
            Range for the width of the Gaussian filter that is used to smooth the spikes.
            Default is (0.05, 0.1).
        
        """
        super(CellParams, self).__init__()
        self.generate_parameters(number_of_cells=number_of_cells, size_mean_and_std=size_mean_and_std,
                                 equal_dimensions=equal_dimensions, coordinates=coordinates,
                                 spikiness_range=spikiness_range,
                                 spike_size_range=spike_size_range,
                                 spike_smoothness_range=spike_smoothness_range)

    def generate_parameters(self, number_of_cells, size_mean_and_std, equal_dimensions, coordinates,
                            spikiness_range, spike_size_range, spike_smoothness_range):
        """
        Generates random cells sizes and rotation angles.

        Parameters:
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

        if not equal_dimensions:
            theta_range = [0, np.pi]
            phi_range = [0, 2 * np.pi]

        cells = pd.DataFrame()
        size_mean, size_std = size_mean_and_std
        for i in range(int(number_of_cells)):
            if equal_dimensions:
                sizex = sizey = sizez = np.random.normal(size_mean, size_std)
                if sizex < 1:
                    sizex = sizey = sizez = 1
                phi = theta = 0
            else:
                sizex = np.random.normal(size_mean, size_std)
                sizey = np.random.normal(size_mean, size_std)
                sizez = np.random.normal(size_mean, size_std)

                if sizex < 1:
                    sizex = 1

                if sizey < 1:
                    sizey = 1

                if sizez < 1:
                    sizez = 1

                theta = self.__sine_distribution(theta_range[0], theta_range[1])
                phi = np.random.uniform(phi_range[0], phi_range[1])

            cell = pd.Series({'size_x': sizex, 'size_y': sizey, 'size_z': sizez, 'phi': phi,
                              'theta': theta})
            if coordinates:
                cell['z'], cell['y'], cell['x'] = np.random.uniform(0, 1, 3)

            if spikiness_range[0] == spikiness_range[1]:
                cell['spikiness'] = spikiness_range[0]
            else:
                cell['spikiness'] = np.random.uniform(spikiness_range[0], spikiness_range[1])
            if spike_size_range[0] == spike_size_range[1]:
                cell['spike_size'] = spike_size_range[0]
            else:
                cell['spike_size'] = np.random.uniform(spike_size_range[0], spike_size_range[1])
            if spike_smoothness_range[0] == spike_smoothness_range[1]:
                cell['spike_smoothness'] = spike_smoothness_range[0]
            else:
                cell['spike_smoothness'] = np.random.uniform(spike_smoothness_range[0], spike_smoothness_range[1])
            cells = cells.append(cell, ignore_index=True)

        for col in cells.columns:
            self[col] = cells[col]

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

    def __sine_distribution(self, minval, maxval, size=1):
        """
        Generates a random values or an array of values in a given range, which is distributed as sin(x) 
    
        Parameters
        ----------
        minval : float
            Minimal value of the range of the random variable.
        maxval : float
            Maximal value of the range of the random variable.
        size : integer, optional
            Number of random values to generate.
            Default: 1
    
        Returns
        -------
        scalar or sequence of scalars
            Returned array of sine-distributed values.
        """
        out = []
        while len(out) < size:  # check whether the target size of the array has been reached
            x = np.random.rand()*(maxval - minval) + minval  # generate a random number between minval and maxval
            y = np.random.rand()  # generate a random number between 0 and 1

            # accept the generated value x, if sin(x) >= y;
            # values around pi/2 will be accepted with high probability,
            # while values around 0 and pi will be accepted with low probability
            if y <= np.sin(x):
                out.append(x)

        if size == 1:
            out = out[0]

        return out

