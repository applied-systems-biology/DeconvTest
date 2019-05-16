from __future__ import division

import numpy as np

from image import Image
from metadata import Metadata
from DeconvTest.modules import input_objects


class Cell(Image):
    """
    Class for a 3D image of a cell.

    """

    def __init__(self, filename=None, ind=None, position=None, input_voxel_size=None, **generate_kwargs):
        """
        Initializes the cell image from a given list of indices or by reading from file.

        Parameters
        ----------
        filename: str, optional
            Path used to load the cell image.
            If None or non-existent, no image will be loaded.
            Default is None.
        ind : ndarray or list, optional
            3D coordinates of all pixels of the binary mask of the cell. 
            The size of the first axis of the ndarray or the length of the list should be equal to 
             the number of dimensions in the cell image (3 for 3D image). 
            The size of the second axis of the ndarray or the length of each sublist corresponds to 
             the number of pixels belonging to the cell mask.
        position : sequence of floats, optional
            Coordinates of the cell center in a multicellular stack in pixels. 
            The length of the sequence should correspond to the number of dimensions in the cell image
             (3 for 3D image).
        input_voxel_size : scalar or sequence of scalars, optional
            Voxel size in z, y and x used to generate the cell image.
            If not None, a cell image will be generated with the give voxel size and cell parameters specified 
             in `generate_kwargs`.
            Default is None.
        generate_kwargs : key, value pairings
            Keyword arguments passed to the `self.generate` function.
        """
        super(Cell, self).__init__(filename=filename)
        self.position = position
        if ind is not None:
            self.from_index(ind)
        elif input_voxel_size is not None:
            self.generate(input_voxel_size=input_voxel_size, **generate_kwargs)

    def __repr__(self):
        if self.position is None:
            return "Cell with unknown position"
        else:
            return "Cell with x=" + str(self.position[2]) \
                   + " y=" + str(self.position[1]) \
                   + " z=" + str(self.position[0]) + " pixels"

    def from_index(self, ind):
        """
        This function initializes the cell from a given list of indices.

        Parameters
        ----------
        ind : ndarray or list, optional
            3D coordinates of all pixels of the binary mask of the cell. 
            The size of the first axis of the ndarray or the length of the list should be equal to 
             the number of dimensions in the cell image (3 for 3D image). 
            The size of the second axis of the ndarray or the length of each sublist corresponds to 
             the number of pixels belonging to the cell mask.
        """
        try:
            ind = np.array(ind)
        except IndexError:
            raise IndexError('number of coordinates in all dimensions must be equal')
        center = np.mean(ind, axis=1)

        center = np.array([np.int_(np.round_(center))]).flatten()

        # compute minimal and maximal indices
        indmin = np.min(ind, axis=1)
        indmax = np.max(ind, axis=1)

        # compute the shape of the cell image
        shape = np.int_(np.round_((np.max([[indmax - center], [center - indmin]], axis=0) * 2 + 5)[0]))

        ncenter = np.int_((shape - 1) / 2)  # compute the image center
        if type(shape) is not np.ndarray:
            shape = np.array([shape])
            ncenter = np.array([ncenter])

        # recompute the coordinates to fit the cell image
        ind = ind - center.reshape((len(center), 1)) + ncenter.reshape((len(ncenter), 1))

        # create an empty array and fill the corresponding coordinates
        self.image = np.zeros(shape)
        ind = np.int_(np.round_(ind))
        self.image[tuple(ind)] = 255

    def generate(self, input_voxel_size, input_cell_kind='ellipsoid', **kwargs):
        """
        Generates a synthetic object image from given parameters and stores the output in the `self.image` variable.

        Parameters
        ----------
        input_voxel_size : scalar or sequence of scalars
            Voxel size in z, y and x used to generate the object image.
            If one value is provided, the voxel size is assumed to be equal along all axes.
        input_cell_kind : string, optional
            Name of the shape of the ground truth object from set of
            {ellipoid, spiky_cell}.
            Default is 'ellipsoid'
        kwargs : key, value pairings
            Keyword arguments passed to corresponding methods to generate synthetic objects.
        """

        if 'generate_' + input_cell_kind in dir(input_objects) and input_cell_kind in input_objects.valid_shapes:
            self.metadata = Metadata()
            self.metadata.set_voxel_size(input_voxel_size)
            self.image = getattr(input_objects, 'generate_' + input_cell_kind)(self.metadata['Voxel size arr'],
                                                                               **kwargs)
            for c in kwargs:
                self.metadata[c] = kwargs[c]
            self.metadata['kind'] = input_cell_kind
            self.metadata['Convolved'] = False
        else:
            raise AttributeError(input_cell_kind + ' is not a valid object shape!')

    def volume(self):
        """
        Computes the volume of the current cell mask (`self.image`) in voxels.
        
        Return
        ------
        float
            The volume of the cell mask in voxels.

        """
        return np.sum((self.image > 0) * 1.)


