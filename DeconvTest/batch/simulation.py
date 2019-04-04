"""
Module containing functions for simulating a microscopy process in a batch mode. 
Includes functions for generating synthetic cells and PSFs, convolution, resizing, and adding noise.
"""
from __future__ import division

import os
import numpy as np

from DeconvTest import CellParams
from DeconvTest import StackParams
from DeconvTest import Cell, Stack, PSF
from DeconvTest.classes.metadata import Metadata
from helper_lib.parallel import run_parallel
from helper_lib import filelib


def generate_cell_parameters(outputfile, number_of_cells=10, size_mean_and_std=(10, 2), equal_dimensions=False,
                             spikiness_range=(0, 0), spike_size_range=(0, 0),
                             spike_smoothness_range=(0.05, 0.1)):
    """
    Generates random cell parameters and save them into a given csv file.
    
    Parameters
    ----------
    outputfile : str
        Path to save cell parameters.
    number_of_cells: int, optional
        Number of cells to generate.
        Default is 1.
    size_mean_and_std: tuple, optional
        Mean value and standard deviation for the cell size in micrometers.
        The cell size is drawn randomly from a Gaussian distribution with the given mean and standard deviation.
        Default is (10, 2).
    equal_dimensions: bool, optional
        If True, generates parameters for a sphere.
        If False, generate parameters for an ellipsoid with sizes for all three axes chosen independently.
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
    if not outputfile.endswith('.csv'):
        outputfile += '.csv'
    params = CellParams(number_of_cells=number_of_cells, size_mean_and_std=size_mean_and_std,
                        equal_dimensions=equal_dimensions, coordinates=False,
                        spikiness_range=spikiness_range,
                        spike_size_range=spike_size_range,
                        spike_smoothness_range=spike_smoothness_range)
    params.save(outputfile)


def generate_cells_batch(params_file, outputfolder, resolution, max_threads=8, print_progress=True):
    """
    Generate synthetic cells with given parameters in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    params_file : str
        Path to a csv file with cell parameters.
    outputfolder : str 
        Output directory to save the generated cells.
    resolution : scalar or sequence of scalars, optional
        Voxel size in z, y and x used to generate the cell image.
        If one value is provided, the voxel size is assume to be equal along all axes.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.

    """
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    if not os.path.exists(params_file):
        raise ValueError('Parameter file does not exist!')
    params = CellParams()
    params.read_from_csv(filename=params_file)
    items = []
    for i in range(len(params)):
        items.append(('cell_%03d.tif' % i, params.iloc[i]))

    kwargs = {'items': items, 'outputfolder': outputfolder, 'resolution': resolution,
              'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__generate_cells_batch_helper, process_name='Generation of cells', **kwargs)


def generate_stack_parameters(outputfolder, number_of_stacks, number_of_cells=10,
                              size_mean_and_std=(10, 2), equal_dimensions=False,
                              spikiness_range=(0, 0), spike_size_range=(0, 0),
                              spike_smoothness_range=(0.05, 0.1)):
    """
    Generate random cell parameters for a given number of multicellular stacks and save them into individual csv files
     in a given directory.
     
    Parameters
    ----------
    outputfolder : str
        Output directory to save the cell parameters.
    number_of_stacks : int
        Number of stacks to generate.
    number_of_cells: int or tuple, optional
        Number of cells in each stack.
        If only one number is provided, the number of cells will be equal in all stacks.
        If a range is provided, the number of cells will be drawn randomly uniformly from the given range.
        Default is 10.
    size_mean_and_std: tuple, optional
        Mean value and standard deviation for the cell size in micrometers.
        The cell size is drawn randomly from a Gaussian distribution with the given mean and standard deviation.
        Default is (10, 2).
    equal_dimensions: bool, optional
        If True, generates parameters for a sphere.
        If False, generate parameters for an ellipsoid with sizes for all three axes chosen independently.
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
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    params = StackParams(number_of_stacks=number_of_stacks, number_of_cells=number_of_cells,
                         size_mean_and_std=size_mean_and_std, equal_dimensions=equal_dimensions,
                         spikiness_range=spikiness_range,
                         spike_size_range=spike_size_range,
                         spike_smoothness_range=spike_smoothness_range)
    params.save(outputfolder)


def generate_stacks_batch(params_folder, outputfolder, resolution, stack_size_microns,
                          max_threads=8, print_progress=True):
    """
    Generate synthetic multicellular stacks with given parameters in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    params_folder : str
        Directory with csv files with cell parameters.
    outputfolder : str 
        Output directory to save the generated stacks.
    resolution : scalar or sequence of scalars, optional
        Voxel size in z, y and x used to generate the stack image.
        If one value is provided, the voxel size is assume to be equal along all axes.
    stack_size_microns : sequence of scalars
        Dimensions of the image stack in micrometers.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.

    """
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    if not params_folder.endswith('/'):
        params_folder += '/'
    if not os.path.exists(params_folder):
        raise ValueError('Parameter folder does not exist!')
    files = filelib.list_subfolders(params_folder, extensions=['csv'])
    kwargs = {'items': files, 'outputfolder': outputfolder, 'resolution': resolution,
              'inputfolder': params_folder, 'stack_size': stack_size_microns,
              'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__generate_stacks_batch_helper, process_name='Generation of stacks', **kwargs)


def generate_psfs_batch(outputfolder, sigmas, elongations, resolution, max_threads=8, print_progress=True):
    """
    Generate synthetic PSFs with given widths in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    outputfolder : str 
        Output directory to save the generated stacks.
    sigmas : sequence of floats
        Standard deviations of the PSF in xy in micrometers.
    elongations : sequence of floats
        PSF aspect ratios.
    resolution : scalar or sequence of scalars, optional
        Voxel size in z, y and x used to generate the PSF image.
        If one value is provided, the voxel size is assume to be equal along all axes.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.

    """
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    items = []
    for sigma in sigmas:
        for elongation in elongations:
            items.append((sigma, elongation))

    kwargs = {'items': items, 'outputfolder': outputfolder, 'resolution': resolution,
              'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__generate_psfs_batch_helper, process_name='Generation of PSFs', **kwargs)


def convolve_batch(inputfolder, psffolder, outputfolder, max_threads=8, print_progress=True):
    """
    Convolves all cell in a given input directory with all PSFs in a given psf directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to convolve.
    psffolder : str
        Directory with PSF images to use for convolution.
    outputfolder : str
        Output directory to save the convolved images.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if not psffolder.endswith('/'):
        psffolder += '/'
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    psffiles = filelib.list_subfolders(psffolder)

    items = [(inputfile, psffile) for inputfile in inputfiles for psffile in psffiles]
    kwargs = {'items': items, 'inputfolder': inputfolder, 'psffolder': psffolder,
              'outputfolder': outputfolder, 'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__convolve_batch_helper, process_name='Convolution', **kwargs)


def resize_batch(inputfolder, outputfolder, resolutions, order=1, max_threads=8, print_progress=True,
                 append_resolution_to_filename=True):
    """
    Resizes all cell images in a given input directory in a parallel mode and saves them in a given output directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to resize.
    outputfolder : str
        Output directory to save the resized images.
    resolutions : list
        List of new voxel sizes to which the input images should be resized.
        Each item of the list is a scalar (for the same voxels size along all axes) 
         or sequence of scalars (voxel size in z, y and x).
    order : int, optional
        The order of the spline interpolation used for resizing. 
        The order has to be in the range 0-5.
        Default is 1.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.
    append_resolution_to_filename : bool, optional
        If True, the information about the new voxel size will be added to the subdirectory name if the image 
         is stored in a subdirectory, or to the image file name if the image is not stored in a subdirectory 
         but in the root directorty.
        If False, a new directory will be created for the corresponding voxel size, an all file and subdirectory
         names will be kept as they are.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    items = [(inputfile, resolution) for inputfile in inputfiles for resolution in resolutions]
    kwargs = {'items': items, 'inputfolder': inputfolder, 'order': order,
              'outputfolder': outputfolder, 'max_threads': max_threads, 'print_progress': print_progress,
              'append_resolution_to_filename': append_resolution_to_filename}
    run_parallel(process=__resize_batch_helper, process_name='Resize', **kwargs)


def add_noise_batch(inputfolder, outputfolder, gaussian_snrs=None, poisson_snrs=None,
                    max_threads=8, print_progress=True):
    """
    Adds synthetic noise to all images in a given input directory in a parallel mode and saves them in a given 
     output directory.
     
    All combination of the given Gaussian and Poisson SNR are generated by first adding the Gaussian noise and then
     adding the Poisson noise.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to which noise should be added.
    outputfolder : str
        Output directory to save the noisy images.
    gaussian_snrs : sequence of floats, optional
        List of target signal-to-noise ratios (SNR) after adding the Gaussian noise.
        None corresponds to no Gaussian noise.
        Default is None.
    poisson_snrs : sequence of floats, optional
        List of target signal-to-noise ratios (SNR) after adding the Poisson noise.
        None corresponds to no Poisson noise.
        Default is None.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    if gaussian_snrs is None:
        gaussian_snrs = [None]
    if poisson_snrs is None:
        poisson_snrs = [None]
    items = [(inputfile, gaus, poisson) for inputfile in inputfiles
             for gaus in gaussian_snrs for poisson in poisson_snrs]
    kwargs = {'items': items, 'inputfolder': inputfolder,
              'outputfolder': outputfolder, 'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__add_noise_batch_helper, process_name='Add noise', **kwargs)


####################################
# private helper functions


def __generate_cells_batch_helper(item, outputfolder, resolution):
    filename, params = item
    cell = Cell(resolution=resolution, **dict(params))
    cell.save(outputfolder + filename)
    cell.metadata.save(outputfolder + filename[:-4] + '.csv')


def __generate_stacks_batch_helper(item, inputfolder, outputfolder, resolution, stack_size):
    params = CellParams()
    params.read_from_csv(filename=inputfolder + item)
    stack = Stack(resolution=resolution, stack_size=stack_size, cell_params=params)
    stack.save(outputfolder + item[:-4] + '.tif')
    stack.metadata.save(outputfolder + item[:-4] + '.csv')


def __generate_psfs_batch_helper(item, outputfolder, resolution):
    metadata = Metadata(resolution)
    resolution = metadata['resolution']
    sigma, elongation = item
    sigmaz = sigma * elongation
    sigmax = sigma / resolution[1]
    sigmaz = sigmaz / resolution[0]
    psf = PSF(sigmax, sigmaz / sigmax)
    psf.save(outputfolder + 'psf_sigma_' + str(sigma) + '_elongation_' + str(elongation) + '.tif',
             normalize_output=True)
    metadata.save(outputfolder + 'psf_sigma_' + str(sigma) + '_elongation_' + str(elongation) + '.csv')


def __convolve_batch_helper(item, inputfolder, psffolder, outputfolder):
    inputfile, psffile = item
    stack = Stack(filename=inputfolder + inputfile)
    psf = PSF(filename=psffolder + psffile)
    stack.convolve(psf)
    stack.save(outputfolder + psffile[:-4] + '/' + inputfile)
    stack.metadata.save(outputfolder + psffile[:-4] + '/' + inputfile[:-4] + '.csv')
    psf.save(outputfolder + psffile)
    stack.metadata.save(outputfolder + psffile[:-4] + '.csv')


def __resize_batch_helper(item, order, inputfolder, outputfolder, append_resolution_to_filename):
    inputfile, resolution = item
    metadata = Metadata(resolution)
    resolution = metadata['resolution']
    stack = Stack(filename=inputfolder + inputfile)
    zoom = np.array(stack.metadata['resolution']) / np.array(resolution)
    stack.resize(zoom=zoom, order=order)
    stack.metadata = metadata
    path = os.path.dirname(inputfile)
    strres = str(resolution).replace('  ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ', '_')
    if path == '':
        if append_resolution_to_filename:
            outputname = outputfolder + inputfile[:-4] + '_voxel_size_' + strres + '.tif'
        else:
            outputname = outputfolder + path + 'voxel_size_' + strres + '/' + inputfile.split('/')[-1]
    else:
        outputname = outputfolder + path + '_voxel_size_' + strres + '/' + inputfile.split('/')[-1]
    stack.save(outputname)
    stack.metadata.save(outputname[:-4] + '.csv')


def __add_noise_batch_helper(item, inputfolder, outputfolder):
    inputfile, gaussian_snr, poisson_snr = item
    stack = Stack(filename=inputfolder + inputfile)
    path = os.path.dirname(inputfile)
    name = inputfile.split('/')[-1]
    if len(name.split('psf')) == 1:
        stack.add_gaussian_noise(gaussian_snr)
        stack.normalize()
        stack.add_poisson_noise(poisson_snr)
        stack.normalize()

    if path == '':
        outputname = outputfolder + inputfile
    else:
        outputname = outputfolder + path + '_gaussian_snr=' + str(gaussian_snr) \
                     + '_poisson_snr=' + str(poisson_snr) + '/' + inputfile.split('/')[-1]
    stack.save(outputname)
    stack.metadata.save(outputname[:-4] + '.csv')
