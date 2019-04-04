from __future__ import division

import os
import numpy as np

from helper_lib import filelib
from cellparams import CellParams


class StackParams(object):
    """
    Class for generation and storing parameters for a set of multicellular stacks.
    """

    def __init__(self, number_of_stacks=1, number_of_cells=1, **kwargs):
        """
        Initializes the class for StackParams and generate random cell parameters with given properties for each stack.
        
        Parameters
        ----------
        number_of_stacks : int, optional
            Number of stacks to generate.
            Default is 1.
        number_of_cells: int or tuple, optional
            Number of cells in each stack.
            If only one number is provided, the number of cells will be equal in all stacks.
            If a range is provided, the number of cells will be drawn randomly uniformly from the given range.
            Default is 1.
        kwargs : key, value pairings
            Keyword arguments passed to the initializer of the CellParams class.
        """
        self.stacks = []
        number_of_cells = np.array([number_of_cells]).flatten()
        for i in range(number_of_stacks):
            if len(number_of_cells) == 1:
                cur_number_of_cells = number_of_cells[0]
            elif len(number_of_cells) == 2:
                cur_number_of_cells = np.random.randint(number_of_cells[0], number_of_cells[1])
            else:
                raise ValueError("Number of cells must be integer or tuple of two integers!")
            self.stacks.append(CellParams(coordinates=True, number_of_cells=cur_number_of_cells, **kwargs))

    def __repr__(self):
        return "Parameters of " + str(len(self.stacks)) + " stacks"

    def save(self, outputfolder):
        """
        Saves the cell parameters of each stack into a csv file in a given directory.

        Parameters
        ----------
        outputfolder: str
            Directory where the cell parameters should be saved.
        """
        if not outputfolder.endswith('/'):
            outputfolder += '/'
        filelib.make_folders([outputfolder])
        for i in range(len(self.stacks)):
            self.stacks[i].save(outputfolder + 'stack%03d.csv' % i)

    def read_from_csv(self, inputfolder):
        """
        Reads cell parameters for multicellular stacks from csv files located in a give directory.
        
        Parameters
        ----------
        inputfolder : str
            Directory with cell parameters. 
            For each csv file in the given directory, a `CellParams` instance will be generated and
             appended to the `self.stacks` list.
        """
        if os.path.exists(inputfolder):
            if not inputfolder.endswith('/'):
                inputfolder += '/'
            files = filelib.list_subfolders(inputfolder, extensions=['csv'])
            self.stacks = []
            for fn in files:
                celldata = CellParams()
                celldata.read_from_csv(inputfolder + fn)
                self.stacks.append(celldata)
