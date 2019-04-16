from __future__ import division

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata
from skimage import morphology

from image import Image
from metadata import Metadata
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

        valid_shapes = ['ellipsoid', 'spiky_cell']

        if 'generate_' + kind in dir(self) and kind in valid_shapes:
            self.metadata = Metadata(resolution=resolution)
            self.image = getattr(self, 'generate_' + kind)(**kwargs)
        else:
            raise AttributeError(kind + ' is not a valid object shape!')

    def generate_ellipsoid(self, size=None, size_x=10, size_y=10, size_z=10, theta=0, phi=0, **kwargs_to_ignore):
        """
        Generates a synthetic object of ellipsoidal shape.

        Parameters
        ----------
        size : scalar or sequence of scalars, optional
            Size of the cell in micrometers.
            If only one value is provided, the size along all axes is assume to be equal (spherical cell).
            To specify individual size for each axis (ellipsoid cells), 3 values should be provided ([z, y, x]).
            If None, the cell sizes in x, y, and z are extracted from `size_x`, `size_y`, `size_z`.
            Default is None.
        size_x, size_y, size_z : scalar or sequence of scalars, optional
            Cell size in micrometers in x, y, and z.
            The values specified here are only used if the parameter `size` is None.
            Default is 10.
        phi : float, optional
            Azimuthal rotation angle in the range from 0 to 2 pi.
            If 0, no azimuthal rotation is done.
            Default is 0.
        theta : float, optional
            Polar rotation angle in the range from 0 to pi.
            If 0, no polar rotation is done.
            Default is 0.
        """

        sizes = self.__convert_size(size, size_x, size_y, size_z)
        sizes = sizes / self.metadata.resolution  # convert the sizes into pixels

        phi = phi * 180 / np.pi  # convert the angles from radians to degrees
        theta = theta * 180 / np.pi

        target_volume = 4. / 3 * np.pi * sizes[0] * sizes[1] * sizes[2] / 8.  # compute the volume of the cell in voxels
        maxsize = np.max([sizes])  # maximal dimension
        side = int(maxsize * 0.6) + 2  # half-dimension of the cell image

        x = np.zeros([side * 2 + 1, side * 2 + 1, side * 2 + 1])  # empty array for the cell image
        x[side, side, side] = 1000.  # a peak in the center
        sig = sizes / 7.5  # compute the optimal sigma for smoothing

        x = ndimage.gaussian_filter(x, sig)  # smooth the peak with a Gaussian filter

        if theta > 0:  # rotate the image, if rotation angles are not 0
            x = ndimage.interpolation.rotate(x, theta, axes=(0, 1))
            if phi > 0:
                x = ndimage.interpolation.rotate(x, phi, axes=(1, 2))

        # compute the intensity percentile (for the thresholding) that corresponds to the target volume of the cells
        per = 100 - target_volume * 100. / (x.shape[0] * x.shape[1] * x.shape[2])
        x = (x >= np.percentile(x, per)) * 255  # threshold the image at the computed intensity percentile

        return x

    def generate_spiky_cell(self, size=None, size_x=10, size_y=10, size_z=10, theta=0, phi=0,
                            spikiness=0.1, spike_size=0.5, spike_smoothness=0.05):
        """
        Generates a synthetic object of ellipsoidal shape.

        Parameters
        ----------
        size : scalar or sequence of scalars, optional
            Size of the cell in micrometers.
            If only one value is provided, the size along all axes is assume to be equal (spherical cell).
            To specify individual size for each axis (ellipsoid cells), 3 values should be provided ([z, y, x]).
            If None, the cell sizes in x, y, and z are extracted from `size_x`, `size_y`, `size_z`.
            Default is None.
        size_x, size_y, size_z : scalar or sequence of scalars, optional
            Cell size in micrometers in x, y, and z.
            The values specified here are only used if the parameter `size` is None.
            Default is 10.
        phi : float, optional
            Azimuthal rotation angle in the range from 0 to 2 pi.
            If 0, no azimuthal rotation is done.
            Default is 0.
        theta : float, optional
            Polar rotation angle in the range from 0 to pi.
            If 0, no polar rotation is done.
            Default is 0.
        spikiness : float in the range [0,1], optional
            Fraction of cell surface area covered by spikes.
            If 0, no spikes will be added.
            If 1, all point of the surface will be replaced by a random radius.
            Default is 0.1.
        spike_size : float, optional
            Standard deviation for the spike amplitude relative to the cell radius.
            If 0, no spikes will be added (amplitude 0).
            Values in the range from 0 to 1 are recommended.
            Default is 0.5.
        spike_smoothness : float, optional
            Width of the Gaussian filter that is used to smooth the spikes.
            The value is provided relative to the cell radius.
            0 corresponds to no Gaussian smoothing.
            Default is 0.05.
        """
        sizes = self.__convert_size(size, size_x, size_y, size_z) / 2

        gridsize = 100
        number_of_spikes = int(spikiness * gridsize**2)
        Phi, Theta = np.meshgrid(np.linspace(0, 2 * np.pi, gridsize, endpoint=True),
                                 np.linspace(0, np.pi, gridsize, endpoint=True))

        x = np.sin(Theta) * np.cos(Phi) * np.cos(theta) + np.cos(Theta) * np.sin(theta)
        y = np.sin(Theta) * np.sin(Phi)
        z = np.cos(Theta) * np.cos(theta) - np.sin(Theta) * np.cos(Phi) * np.sin(theta)
        grid = np.sqrt(1 / ((x / sizes[0]) ** 2 + (y / sizes[1]) ** 2 + (z / sizes[2]) ** 2))

        if phi > 0:
            i = np.argmin(abs(Phi[1] - phi))
            if i > 0:
                R = np.zeros_like(grid)
                R[:, :i] = grid[:, -i:]
                R[:, i:] = grid[:, :-i]
                grid = R

        # introduce spikes

        phi = np.random.randint(0, grid.shape[1], size=number_of_spikes)
        theta = []
        while len(theta) < number_of_spikes:
            x = np.random.randint(0, grid.shape[0])
            y = np.random.rand()
            if y <= np.sin(Theta[x, 0]):
                theta.append(x)

        if spike_size > 0:
            random_points = np.random.normal(1, spike_size, size=number_of_spikes)
            grid[theta, phi] = grid[theta, phi] * random_points
            if spike_smoothness > 0:
                grid = ndimage.gaussian_filter(grid, spike_smoothness * gridsize / np.pi)
            grid[grid < 0] = 0

        # interpolate the grid
        points = np.array([Theta.flatten(), Phi.flatten()]).transpose()
        Phi, Theta = np.meshgrid(np.linspace(0, 2 * np.pi, 500, endpoint=True),
                                 np.linspace(0, np.pi, 500, endpoint=True))

        xi = np.asarray([[Theta[i, j], Phi[i, j]] for i in range(len(Theta)) for j in range(len(Theta[0]))])
        grid = griddata(points, grid.flatten(), xi, method='linear')
        grid = grid.reshape((500, 500))

        # fill in the cell interior
        Grid = [grid]
        n = int(round(np.max(sizes / self.metadata.resolution)))
        for i in range(n):
            Grid.append(i * grid / n)

        # convert to Cartesian coordinates and make an image
        img = None
        mincoords = None
        for grid in Grid:
            coords = self.__spherical_to_cart(grid, Phi, Theta)
            coords = np.array(coords).reshape([3, len(coords[0])])
            for i in range(len(coords)):
                coords[i] = np.int_(np.round_(coords[i] / self.metadata.resolution[i]))
            if mincoords is None:
                mincoords = np.min(coords, axis=1)

            for i in range(len(coords)):
                coords[i] = coords[i] - mincoords[i] + 5

            coords = np.int_(coords)
            if img is None:
                img = np.zeros(np.max(coords, axis=1) + 5)  # empty array for the cell image
            z, y, x = coords
            img[z, y, x] = 255

        img = (morphology.remove_small_objects((img > 0).astype(bool),
                                               min_size=4. / 3 * np.pi * sizes[0] * sizes[1] * sizes[2] / 2) > 0) * 255

        return img

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

    #########################################################
    # private helper functions

    def __spherical_to_cart(self, r, phi, theta):
        x = r * np.sin(phi) * np.cos(theta - np.pi)
        y = r * np.sin(phi) * np.sin(theta - np.pi)
        z = r * np.cos(phi)
        return x.flatten(), y.flatten(), z.flatten()

    def __convert_size(self, size=None, size_x=10, size_y=10, size_z=10):
        if size is not None:
            size = np.array([size]).flatten()
            if len(size) == 1:
                sizes = np.array([size[0], size[0], size[0]])
            elif len(size) == 3:
                sizes = size
            else:
                raise ValueError('The size value has to be a number or sequence of length 3!')
        else:
            sizes = np.array([size_z, size_y, size_x])
        return sizes



