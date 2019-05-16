from __future__ import division

import numpy as np
from scipy import ndimage


from cell import Cell
from image import Image
from metadata import Metadata


class Stack(Image):
    """
    Class for a 3D multicellular stack.
    """

    def __init__(self, filename=None, is_segmented=False, is_labeled=False,
                 input_voxel_size=None, stack_size=None, cell_params=None):
        """
        Initializes the Stack class.

        Parameters
        ----------
        filename: str, optional
            Path used to load the cell image.
            If None or non-existent, no image will be loaded.
            Default is None.
        is_segmented : bool, optional
            If True the cell stack is assumed to be already segmented (binary mask or array of labeled objects).
            If False the cell stack is assumed to be not segmented yet.
            Default is False.
        is_labeled : bool, optional
            If True it is assumed that individual connected regions are labeled by unique labels.
            If False individual cells are assumed not to be labeled.
            Default is False.
        input_voxel_size : scalar or sequence of scalars, optional
            Voxel size used to generate the stack.
            If None, no stack will be generated.
            Default is None.
        stack_size : sequence of scalars, optional
            Dimensions of the image stack in micrometers.
            If None, no stack will be generated.
            Default is None.
        cell_params : pandas.DataFrame or CellParams
            Dictionary of cell parameters.
            The columns should include the keyword arguments passed though `Cell.generate`.
            If None, no stack will be generated.
            Default is None.
        """
        super(Stack, self).__init__(filename=filename)
        self.cells = []
        self.is_segmented = is_segmented
        self.is_labeled = is_labeled
        if input_voxel_size is not None and stack_size is not None and cell_params is not None:
            self.generate(cell_params, input_voxel_size, stack_size)

    def __repr__(self):
        return "Stack with " + str(len(self.cells)) + " cells"

    def generate(self, cell_params, input_voxel_size, stack_size):
        """
        Generates a multicellular stack image from cell parameters.

        Parameters
        ----------
        cell_params : pandas.DataFrame or CellParams
            Dictionary of cell parameters.
            The columns should include the keyword arguments passed though `Cell.generate`.
        input_voxel_size : scalar or sequence of scalars
            Voxel size used to generate the stack.
        stack_size : sequence of scalars
            Dimensions of the image stack in micrometers.
        """
        stack_size_pix = np.int_(np.round_(np.array(stack_size) / np.array(input_voxel_size)))
        if 'size' not in cell_params and \
                ('size_x' not in cell_params or 'size_y' not in cell_params or 'size_z' not in cell_params):
            raise ValueError('Either `size` or `size_x`, `size_y` and `size_z` must be provided!')
        if 'size' in cell_params:
            shift = np.array([np.mean(cell_params['size'])]*3)
        else:
            shift = np.array([cell_params['size_z'].mean(), cell_params['size_y'].mean(), cell_params['size_x'].mean()])
        shift = shift / np.array(input_voxel_size) / 2 / stack_size_pix
        for i, c in enumerate(['z', 'y', 'x']):
            cell_params.loc[:, c] = np.array(cell_params[c])*(1 - 2*shift[i]) + shift[i]
        self.image = np.zeros(stack_size_pix)

        for i in range(len(cell_params)):
            p = dict(cell_params.iloc[i])
            x = p.pop('x', None)
            y = p.pop('y', None)
            z = p.pop('z', None)
            c = Cell(input_voxel_size=input_voxel_size, **p)
            if x is None or y is None or z is None:  # position the cell randomly if the positions are not specified
                c.position = np.random.rand(3) * stack_size_pix
            else:
                if z < 0 or z > 1 or y < 0 or y > 1 or x < 0 or x > 1:
                    raise ValueError('Coordinate values must be in the range from 0 to 1!')
                c.position = np.array([z, y, x]) * stack_size_pix

            self.position_cell(c)
        self.is_segmented = True
        self.is_labeled = False
        self.metadata = Metadata()
        self.metadata.set_voxel_size(input_voxel_size)
        self.metadata['Convolved'] = False
        self.metadata['Number of cells'] = len(cell_params)
        self.metadata['Stack size um'] = stack_size

    def position_cell(self, cell):
        """
        Adds a cell to the current stack at a given position.

        Parameters
        ----------
        cell : Cell
            Cell object to be positioned. 
            The coordinates for cell positioning are given by `cell.position`.
        """
        stack_size = np.array(self.image.shape)
        ind = np.array(np.where(cell.image > 0))  # extract the cell coordinates
        center = np.array(ndimage.center_of_mass(cell.image)).reshape([3, 1])  # extract the cell center
        ncenter = np.array(cell.position).reshape([3, 1])  # the new cell center is the position in the stack
        cdiff = np.int_(np.round_(ncenter - center))
        ind = ind + cdiff  # recompute the cell coordinates

        # make sure that the new cell coordinates are not outside of the borders of the image stack
        ind_ind = np.where((ind[0] >= 0) & (ind[1] >= 0) & (ind[2] >= 0) & (ind[0] < stack_size[0])
                           & (ind[1] < stack_size[1]) & (ind[2] < stack_size[2]))
        ind = (ind[0][ind_ind], ind[1][ind_ind], ind[2][ind_ind])

        self.image[ind] = 255  # add the new cell to the stack









