from __future__ import division

import numpy as np
import pandas as pd

from image import Image
from metadata import Metadata
from DeconvTest.modules import input_objects
from helper_lib.image import unify_shape, segment


class Cell(Image):
    """
    Class for a 3D image of a cell.

    """

    def __init__(self, filename=None, ind=None, position=None, resolution=None, **generate_kwargs):
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
        resolution : scalar or sequence of scalars, optional
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
        elif resolution is not None:
            self.generate(resolution=resolution, **generate_kwargs)

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

    def generate(self, resolution, kind='ellipsoid', **kwargs):
        """
        Generates a synthetic object image from given parameters and stores the output in the `self.image` variable.

        Parameters
        ----------
        resolution : scalar or sequence of scalars
            Voxel size in z, y and x used to generate the object image.
            If one value is provided, the voxel size is assumed to be equal along all axes.
        kind : string, optional
            Name of the shape of the ground truth object from set of
            {ellipoid, spiky_cell}.
            Default is 'ellipsoid'
        kwargs : key, value pairings
            Keyword arguments passed to corresponding methods to generate synthetic objects.
        """

        if 'generate_' + kind in dir(input_objects) and kind in input_objects.valid_shapes:
            self.metadata = Metadata(resolution=resolution)
            self.image = getattr(input_objects, 'generate_' + kind)(self.metadata.resolution, **kwargs)
        else:
            raise AttributeError(kind + ' is not a valid object shape!')

    def volume(self):
        """
        Computes the volume of the current cell mask (`self.image`) in voxels.
        
        Return
        ------
        float
            The volume of the cell mask in voxels.

        """
        return np.sum((self.image > 0) * 1.)

    def segment(self, preprocess=False, thr=None, relative_thr=False, postprocess=False):
        """
        Segments the current image by thresholding with optional preprocessing.
        
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

        Returns
        -------
        ndarray
            Segmented binary mask.
        """

        if self.image is None:
            raise ValueError('Image is None!')

        if preprocess:
            median = 3
        else:
            median = None
        if postprocess:
            morphology = True
            fill_holes = True
        else:
            morphology = False
            fill_holes = False

        self.image = (segment(self.image, thr=thr, relative_thr=relative_thr, median=median,
                             morphology=morphology, fill_holes=fill_holes, label=False) > 0)*255

        return self.image

    def dimensions(self):
        """
        Computes the size of the object represented by the current binary mask.
        
        Returns
        -------
        sequence of int
            Dimensions / sizes of the object in pixels along all axes.
        """
        if self.image is None:
            raise ValueError
        if len(np.unique(self.image)) > 2:
            raise ValueError('Cannot measure dimensions: image is not binary!')
        dim = []
        ind = np.where(self.image > 0)
        for i in ind:
            if len(i) > 0:
                dim.append(np.max(i) - np.min(i) + 1)
            else:
                dim.append(0)
        return np.array(dim)

    def compare_to_ground_truth(self, gt):
        """
        Computes the overlap errors, Jaccard index, and other accuracy measures between each connected region 
         in the current image and a given ground truth image.
         
        Parameters
        ----------
        gt : Image or Cell 
            Binary ground truth image.

        Returns
        -------
        pandas.Series()
            Dictionary containing the values for the computed accuracy measures.
        """
        data = pd.Series()
        self.image, gt.image = unify_shape(self.image, gt.image)  # convert cell images to the same shape
        self.image = (self.image > 0) * 1.
        gt.image = (gt.image > 0) * 1.
        overlap = np.sum(self.image * gt.image)
        union = np.sum((self.image + gt.image) > 0)
        a = np.sum(gt.image)
        b = np.sum(self.image)
        data['Overdetection error'] = (b - overlap) / a
        data['Underdetection error'] = (a - overlap) / a
        data['Overlap error'] = (union - overlap) / a
        data['Jaccard index'] = overlap / union
        data['Sensitivity'] = overlap / a
        data['Precision'] = overlap / b
        return data






