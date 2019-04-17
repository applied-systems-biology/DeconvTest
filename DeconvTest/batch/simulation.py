"""
Module containing functions for simulating a microscopy process in a batch mode. 
Includes functions for generating synthetic cells and PSFs, convolution, resizing, and adding noise.
"""
from __future__ import division

import os
import numpy as np
import itertools

from DeconvTest import CellParams
from DeconvTest import StackParams
from DeconvTest import Cell, Stack, PSF
from DeconvTest.classes.metadata import Metadata
from helper_lib.parallel import run_parallel
from helper_lib import filelib


def generate_cell_parameters(outputfile, **kwargs):
    """
    Generates random cell parameters and save them into a given csv file.
    
    Parameters
    ----------
    outputfile : str
        Path to save cell parameters.
    kwargs : key, value pairings
        Keyword arguments passed to the `CellParams` class.

    """
    if not outputfile.endswith('.csv'):
        outputfile += '.csv'
    params = CellParams(**kwargs)
    params.save(outputfile)


def generate_cells_batch(params_file, outputfolder, **kwargs):
    """
    Generate synthetic cells with given parameters in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    params_file : str
        Path to a csv file with cell parameters.
    outputfolder : str 
        Output directory to save the generated cells.

    Keyword arguments
    -----------------
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

    kwargs['items'] = items
    kwargs['outputfolder'] = outputfolder
    run_parallel(process=__generate_cells_batch_helper, process_name='Generation of cells', **kwargs)


def generate_stack_parameters(outputfolder, **kwargs):
    """
    Generate random cell parameters for a given number of multicellular stacks and save them into individual csv files
     in a given directory.
     
    Parameters
    ----------
    outputfolder : str
        Output directory to save the cell parameters.
    kwargs : key, value pairings
        Keyword arguments passed to the `StackParams` class.
    """
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    params = StackParams(**kwargs)
    params.save(outputfolder)


def generate_stacks_batch(params_folder, outputfolder, **kwargs):
    """
    Generate synthetic multicellular stacks with given parameters in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    params_folder : str
        Directory with csv files with cell parameters.
    outputfolder : str 
        Output directory to save the generated stacks.

    Keyword arguments
    -----------------
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
    kwargs['items'] = files
    kwargs['outputfolder'] = outputfolder
    kwargs['inputfolder'] = params_folder
    run_parallel(process=__generate_stacks_batch_helper, process_name='Generation of stacks', **kwargs)


def generate_psfs_batch(outputfolder, sigmas, aspect_ratios, **kwargs):
    """
    Generate synthetic PSFs with given widths in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    outputfolder : str 
        Output directory to save the generated stacks.
    sigmas : sequence of floats
        Standard deviations of the PSF in xy in micrometers.
    aspect_ratios : sequence of floats
        PSF aspect ratios.

    Keyword arguments
    -----------------
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
        for aspect_ratio in aspect_ratios:
            items.append((sigma, aspect_ratio))
    kwargs['items'] = items
    kwargs['outputfolder'] = outputfolder

    run_parallel(process=__generate_psfs_batch_helper, process_name='Generation of PSFs', **kwargs)


def convolve_batch(inputfolder, psffolder, outputfolder, **kwargs):
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

    Keyword arguments
    -----------------
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
    kwargs['items'] = items
    kwargs['outputfolder'] = outputfolder
    kwargs['inputfolder'] = inputfolder
    kwargs['psffolder'] = psffolder
    run_parallel(process=__convolve_batch_helper, process_name='Convolution', **kwargs)


def resize_batch(inputfolder, outputfolder, resolutions, **kwargs):
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

    Keyword arguments
    -----------------
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
         Default is True.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    items = [(inputfile, resolution) for inputfile in inputfiles for resolution in resolutions]
    kwargs['items'] = items
    kwargs['outputfolder'] = outputfolder
    kwargs['inputfolder'] = inputfolder
    run_parallel(process=__resize_batch_helper, process_name='Resize', **kwargs)


def add_noise_batch(inputfolder, outputfolder, kind, snr, test_snr_combinations=False, **kwargs):
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
    kind : string, sequence of strings or None
        Name of the method to generate nose from set of {gaussian, poisson}.
        If a sequence is provided, several noise types will be added.
        If None, no noise will be added.
    snr : float or sequence of floats
        Target signal-to-noise ratio(s) (SNR) for each noise type.
        If None, no noise is added.
    test_snr_combinations : bool
        If True and several noise types in the `kind` argument are provided, all combinations of the values
         provided in `snr` will be tested for each noise type.

    Keyword arguments
    -----------------
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

    kind = np.array([kind]).flatten()
    snr = np.array([snr]).flatten()
    if len(kind) > 1 and test_snr_combinations:
        snr_items = list(itertools.product(*[snr]*len(kind)))
        items = [(inputfile, kind, list(snr_item)) for inputfile in inputfiles for snr_item in snr_items]

    else:
        items = [(inputfile, kind, snr1) for inputfile in inputfiles for snr1 in snr]

    kwargs['items'] = items
    kwargs['outputfolder'] = outputfolder
    kwargs['inputfolder'] = inputfolder
    run_parallel(process=__add_noise_batch_helper, process_name='Add noise', **kwargs)


####################################
# private helper functions


def __generate_cells_batch_helper(item, outputfolder, resolution, **kwargs_to_ignore):
    filename, params = item
    cell = Cell(resolution=resolution, **dict(params))
    cell.save(outputfolder + filename)
    cell.metadata.save(outputfolder + filename[:-4] + '.csv')


def __generate_stacks_batch_helper(item, inputfolder, outputfolder, resolution,
                                   stack_size_microns, **kwargs_to_ignore):
    params = CellParams()
    params.read_from_csv(filename=inputfolder + item)
    stack = Stack(resolution=resolution, stack_size=stack_size_microns, cell_params=params)
    stack.save(outputfolder + item[:-4] + '.tif')
    stack.metadata.save(outputfolder + item[:-4] + '.csv')


def __generate_psfs_batch_helper(item, outputfolder, resolution, **kwargs_to_ignore):
    metadata = Metadata(resolution)
    resolution = metadata['resolution']
    sigma, elongation = item
    sigmaz = sigma * elongation
    sigmax = sigma / resolution[1]
    sigmaz = sigmaz / resolution[0]
    psf = PSF(sigma=sigmax, aspect_ratio=sigmaz / sigmax)
    psf.save(outputfolder + 'psf_sigma_' + str(sigma) + '_aspect_ratio_' + str(elongation) + '.tif',
             normalize_output=True)
    metadata.save(outputfolder + 'psf_sigma_' + str(sigma) + '_aspect_ratio_' + str(elongation) + '.csv')


def __convolve_batch_helper(item, inputfolder, psffolder, outputfolder, **kwargs_to_ignore):
    inputfile, psffile = item
    stack = Stack(filename=inputfolder + inputfile)
    psf = PSF(filename=psffolder + psffile)
    stack.convolve(psf)
    stack.save(outputfolder + psffile[:-4] + '/' + inputfile)
    stack.metadata.save(outputfolder + psffile[:-4] + '/' + inputfile[:-4] + '.csv')
    psf.save(outputfolder + psffile)
    stack.metadata.save(outputfolder + psffile[:-4] + '.csv')


def __resize_batch_helper(item, inputfolder, outputfolder, order=1, append_resolution_to_filename=True, **kwargs_to_ignore):
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


def __add_noise_batch_helper(item, inputfolder, outputfolder, **kwargs_to_ignore):
    inputfile, kind, snr = item
    stack = Stack(filename=inputfolder + inputfile)
    path = os.path.dirname(inputfile)
    name = inputfile.split('/')[-1]
    if len(name.split('psf')) == 1:
        stack.add_noise(kind=kind, snr=snr)

    if path == '':
        outputname = outputfolder + inputfile
    else:
        outputname = outputfolder + path + '_noise'
        snr = np.array([snr]).flatten()
        if len(snr) == 1:
            snr = np.array([snr[0]] * len(kind))
        for i, k in enumerate(kind):
            outputname = outputname + '_' + k + '_snr=' + str(snr[i])
        outputname = outputname + '/' + inputfile.split('/')[-1]
    stack.save(outputname)
    stack.metadata.save(outputname[:-4] + '.csv')
