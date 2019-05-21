"""
This module contains functions for generating synthetic objects of various shapes
"""

import pandas as pd
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from skimage import morphology

valid_shapes = ['ellipsoid', 'spiky_cell']


def parameters_ellipsoid(size_mean_and_std=(10, 2), equal_dimensions=False, **kwargs_to_ignore):
    """
    Generates random cells sizes and rotation angles.

    Parameters:
    -----------
    size_mean_and_std: tuple, optional
        Mean value and standard deviation for the cell size in micrometers.
        The cell size is drawn randomly from a Gaussian distribution with given mean and standard deviation.
        Default is (10, 2).
    equal_dimensions: bool, optional
        If True, generates parameters for a sphere.
        If False, generate parameters for an ellipsoid with sizes for all three axes chosen independently.
        Default is True
    """

    size_mean, size_std = size_mean_and_std
    if equal_dimensions:
        sizex = sizey = sizez = np.random.normal(size_mean, size_std)
        if sizex < 1:
            sizex = sizey = sizez = 1
        phi = theta = 0
    else:
        theta_range = [0, np.pi]
        phi_range = [0, 2 * np.pi]
        sizex = np.random.normal(size_mean, size_std)
        sizey = np.random.normal(size_mean, size_std)
        sizez = np.random.normal(size_mean, size_std)

        if sizex < 1:
            sizex = 1

        if sizey < 1:
            sizey = 1

        if sizez < 1:
            sizez = 1

        theta = __sine_distribution(theta_range[0], theta_range[1])
        phi = np.random.uniform(phi_range[0], phi_range[1])

    cell = pd.DataFrame({'size_x': [sizex], 'size_y': [sizey], 'size_z': [sizez], 'phi': [phi], 'theta': [theta]})

    return cell


def generate_ellipsoid(input_voxel_size, size=None, size_x=10, size_y=10, size_z=10, theta=0, phi=0,
                       **kwargs_to_ignore):
    """
    Generates a synthetic object of ellipsoidal shape.

    Parameters
    ----------
    input_voxel_size : scalar or sequence of scalars
        Voxel size in z, y and x used to generate the object image.
        If one value is provided, the voxel size is assumed to be equal along all axes.
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

    sizes = __convert_size(size, size_x, size_y, size_z)
    sizes = sizes / np.array(input_voxel_size)  # convert the sizes into pixels

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


def parameters_spiky_cell(size_mean_and_std=(10, 2), equal_dimensions=False,
                          spikiness_range=(0.1, 0.5), spike_size_range=(0.1, 0.5),
                          spike_smoothness_range=(0.05, 0.1), **kwargs_to_ignore):
    """
    Generates random cells sizes and rotation angles.

    Parameters:
    -----------
    size_mean_and_std: tuple, optional
        Mean value and standard deviation for the cell size in micrometers.
        The cell size is drawn randomly from a Gaussian distribution with given mean and standard deviation.
        Default is (10, 2).
    equal_dimensions: bool, optional
        If True, generates parameters for a sphere.
        If False, generate parameters for an ellipsoid with sizes for all three axes chosen independently.
        Default is True
    spikiness_range : tuple, optional
        Range for the fraction of cell surface area covered by spikes.
        Default is (0.1, 0.5).
    spike_size_range : tuple, optional
        Range for the standard deviation for the spike amplitude relative to the cell radius.
        Default is (0.1, 0.5).
    spike_smoothness_range : tuple, optional
        Range for the width of the Gaussian filter that is used to smooth the spikes.
        Default is (0.05, 0.1).

    """

    cell = parameters_ellipsoid(size_mean_and_std=size_mean_and_std, equal_dimensions=equal_dimensions)

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

    return cell


def generate_spiky_cell(input_voxel_size, size=None, size_x=10, size_y=10, size_z=10, theta=0, phi=0,
                        spikiness=0.1, spike_size=0.5, spike_smoothness=0.05, **kwargs_to_ignore):
    """
    Generates a synthetic object of ellipsoidal shape.

    Parameters
    ----------
    input_voxel_size : scalar or sequence of scalars
        Voxel size in z, y and x used to generate the object image.
        If one value is provided, the voxel size is assumed to be equal along all axes.
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
    sizes = __convert_size(size, size_x, size_y, size_z) / 2

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
    n = int(round(np.max(sizes / np.array(input_voxel_size))))
    for i in range(n):
        Grid.append(i * grid / n)

    # convert to Cartesian coordinates and make an image
    img = None
    mincoords = None
    for grid in Grid:
        coords = __spherical_to_cart(grid, Phi, Theta)
        coords = np.array(coords).reshape([3, len(coords[0])])
        for i in range(len(coords)):
            coords[i] = np.int_(np.round_(coords[i] / input_voxel_size[i]))
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


#################################################################

def __sine_distribution(minval, maxval, size=1):
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
        x = np.random.rand() * (maxval - minval) + minval  # generate a random number between minval and maxval
        y = np.random.rand()  # generate a random number between 0 and 1

        # accept the generated value x, if sin(x) >= y;
        # values around pi/2 will be accepted with high probability,
        # while values around 0 and pi will be accepted with low probability
        if y <= np.sin(x):
            out.append(x)

    if size == 1:
        out = out[0]

    return out


def __spherical_to_cart(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta - np.pi)
    y = r * np.sin(phi) * np.sin(theta - np.pi)
    z = r * np.cos(phi)
    return x.flatten(), y.flatten(), z.flatten()


def __convert_size(size=None, size_x=10, size_y=10, size_z=10):
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
