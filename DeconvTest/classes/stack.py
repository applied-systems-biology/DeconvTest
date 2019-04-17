from __future__ import division

import warnings
import numpy as np
import pandas as pd
from scipy import ndimage

from skimage.measure import label as lbl

from cell import Cell
from image import Image
from metadata import Metadata
from DeconvTest.modules import quantification

from helper_lib.image import unify_shape


class Stack(Image):
    """
    Class for a 3D multicellular stack.
    """

    def __init__(self, filename=None, is_segmented=False, is_labeled=False,
                 resolution=None, stack_size=None, cell_params=None):
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
        resolution : scalar or sequence of scalars, optional
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
        if resolution is not None and stack_size is not None and cell_params is not None:
            self.generate(cell_params, resolution, stack_size)

    def __repr__(self):
        return "Stack with " + str(len(self.cells)) + " cells"

    def generate(self, cell_params, resolution, stack_size):
        """
        Generates a multicellular stack image from cell parameters.

        Parameters
        ----------
        cell_params : pandas.DataFrame or CellParams
            Dictionary of cell parameters.
            The columns should include the keyword arguments passed though `Cell.generate`.
        resolution : scalar or sequence of scalars
            Voxel size used to generate the stack.
        stack_size : sequence of scalars
            Dimensions of the image stack in micrometers.
        """
        stack_size_pix = np.int_(np.round_(np.array(stack_size) / np.array(resolution)))
        if 'size' not in cell_params and \
                ('size_x' not in cell_params or 'size_y' not in cell_params or 'size_z' not in cell_params):
            raise ValueError('Either `size` or `size_x`, `size_y` and `size_z` must be provided!')
        if 'size' in cell_params:
            shift = np.array([np.mean(cell_params['size'])]*3)
        else:
            shift = np.array([cell_params['size_z'].mean(), cell_params['size_y'].mean(), cell_params['size_x'].mean()])
        shift = shift / np.array(resolution) / 2 / stack_size_pix
        for i, c in enumerate(['z', 'y', 'x']):
            cell_params.loc[:, c] = np.array(cell_params[c])*(1 - 2*shift[i]) + shift[i]
        self.image = np.zeros(stack_size_pix)

        for i in range(len(cell_params)):
            # create an instance of the CellSimulation class for each line in cell_params
            p = dict(cell_params.iloc[i])
            x = p.pop('x', None)
            y = p.pop('y', None)
            z = p.pop('z', None)
            c = Cell(resolution=resolution, **p)
            if x is None or y is None or z is None:  # position the cell randomly if the positions are not specified
                c.position = np.random.rand(3) * stack_size_pix
            else:
                if z < 0 or z > 1 or y < 0 or y > 1 or x < 0 or x > 1:
                    raise ValueError('Coordinate values must be in the range from 0 to 1!')
                c.position = np.array([z, y, x]) * stack_size_pix

            self.position_cell(c)
        self.is_segmented = True
        self.is_labeled = False
        self.metadata = Metadata(resolution=resolution)

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

    def segment(self, preprocess=False, thr=None, relative_thr=False,
                postprocess=False, label=True, **kwargs_to_ignore):
        """
        Segments the current image by thresholding with optional preprocessing and finding connected regions.
        
        Parameters
        ----------
        preprocess : bool, optional
            If True, the image will be preprocessed with a median filter (size 3) prior to segmentation.
            Default is False.
        thr : scalar, optional
            Threshold value for image segmentation.
            If None, automatic Otsu threshold will be computed.
            Default is None.
        relative_thr : bool, optional
            If True, the value of `thr` is multiplied by the maximum intensity of the image.
            Default is False.
        postprocess bool, optional
            If True, morphological opening and closing and binary holes filling will be applied after theresholding.
            Default is False.
        label : bool, optional
            If True, connected region will be labeled by unique labels.
            Default is True.

        Returns
        -------
        ndarray
            Segmented binary mask or labeled image.
        """
        self.image = quantification.segment(self.image, preprocess, thr, relative_thr, postprocess, label)
        self.is_segmented = True

        if label is True:
            self.is_labeled = True
            self.split_to_cells()
        else:
            self.image = (self.image > 0)*255

        return self.image

    def split_to_cells(self):
        """
        Creates an instance of the `Cell` class for each connected region in the current image.

        """
        if self.image is None:
            raise ValueError('Cannot split into cells: image is None')
        if not self.is_segmented:
            raise ValueError('Cannot split into cells: image is not segmented')
        if not self.is_labeled:
            self.image = lbl(self.image)
            self.is_labeled = True

        llist = np.unique(self.image)
        llist = llist[llist > 0]
        centers = np.round_(ndimage.center_of_mass(self.image, self.image, llist), 1)
        for i, l in enumerate(llist):
            c = Cell(ind=np.where(self.image == l), position=centers[i])
            self.cells.append(c)

    def dimensions(self, **kwargs):
        """
        Computes the sizes of the objects represented by individual connected regions in the current image.
        
        Parameters
        ----------
        kwargs : key, value pairings
            Keyword arguments passed to the `dimensions` function of the `Cell` class.
        
        Returns
        -------
        ndarray
            NxM array of the object sizes. 
            N is the number of individual connected regions.
            M is the number of dimensions in the image (M=3 for 3D).

        """
        if len(self.cells) == 0:
            warnings.warn("No cells were found in the stack!", Warning)
            dims = [np.zeros(3)]
        else:
            dims = []
            for c in self.cells:
                dims.append(c.dimensions(**kwargs))
        return np.int_(np.round_(dims))

    def compute_binary_accuracy_measures(self, gt):
        """   
        Computes the overlap errors, Jaccard index, and other accuracy measures between each connected region 
         in the current image and a given ground truth image.
         
        Parameters
        ----------
        gt : Image or Cell 
            Ground truth image.            

        Returns
        -------
        pandas.DataFrame()
            Dictionary with the computed accuracy measures for all individual connected regions.
            The length of the data frame equals to the number of connected regions in the ground truth image.

        """
        self.image, gt.image = unify_shape(self.image, gt.image)  # convert cell images to the same shape

        data = pd.DataFrame()
        if len(gt.cells) == 0:
            gt.segment()
            if len(gt.cells) == 0:
                raise ValueError("No cells were found in the ground truth stack")

        if len(self.cells) == 0:
            warnings.warn("No cells were found in the stack!", Warning)
            data = pd.DataFrame({'CellID': np.arange(len(gt.cells)),
                                 'Overlap error': np.ones(len(gt.cells)),
                                 'Overdetection error': np.zeros(len(gt.cells)),
                                 'Underdetection error': np.ones(len(gt.cells)),
                                 'Jaccard index': np.zeros(len(gt.cells)),
                                 'Sensitivity': np.zeros(len(gt.cells)),
                                 'Precision': np.zeros(len(gt.cells))})
        else:
            centers = []
            for c in self.cells:
                centers.append(c.position)

            for i in range(len(gt.cells)):  # compare each cell to the closest one in the ground truth Stack
                center = np.reshape(gt.cells[i].position, (1, 3))
                dist = np.sum((centers - center) ** 2, axis=1)
                curdata = self.cells[dist.argmin()].compute_binary_accuracy_measures(gt.cells[i])
                curdata['CellID'] = i
                data = pd.concat([data, curdata], ignore_index=True)

        return data








