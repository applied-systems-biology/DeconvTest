"""
Module containing functions for simulating a microscopy process in a batch mode. 
Includes functions for generating synthetic cells and PSFs, convolution, resizing, and adding noise.
"""
from __future__ import division

import os
import numpy as np
import itertools

from DeconvTest import CellParams
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


def generate_cells_batch(params_file, **kwargs):
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
    if not os.path.exists(params_file):
        raise ValueError('Parameter file does not exist!')
    params = CellParams()
    params.read_from_csv(filename=params_file)
    if 'stack' in params.columns:
        items = []
        for st in params['stack'].unique():
            items.append(('stack_%03d.tif' % st, params[params['stack'] == st]))
        kwargs['items'] = items
        run_parallel(process=__generate_stacks_batch_helper, process_name='Generation of stacks', **kwargs)

    else:
        items = []
        for i in range(len(params)):
            items.append(('cell_%03d.tif' % i, params.iloc[i]))

        kwargs['items'] = items
        run_parallel(process=__generate_cells_batch_helper, process_name='Generation of cells', **kwargs)


def generate_psfs_batch(outputfolder, psf_sigmas=None, psf_aspect_ratios=None, **kwargs):
    """
    Generate synthetic PSFs with given widths in a parallel mode and saves them in a given directory.
    
    Parameters
    ----------
    outputfolder : str 
        Output directory to save the generated stacks.
    psf_sigmas : sequence of floats
        Standard deviations of the PSF in xy in micrometers.
    psf_aspect_ratios : sequence of floats
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
    if psf_sigmas is not None and psf_aspect_ratios is not None:
        for sigma in psf_sigmas:
            for aspect_ratio in psf_aspect_ratios:
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


def resize_batch(inputfolder, outputfolder, voxel_sizes_for_resizing, **kwargs):
    """
    Resizes all cell images in a given input directory in a parallel mode and saves them in a given output directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to resize.
    outputfolder : str
        Output directory to save the resized images.
    voxel_sizes_for_resizing : list
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
    items = [(inputfile, resolution) for inputfile in inputfiles for resolution in voxel_sizes_for_resizing]
    kwargs['items'] = items
    kwargs['outputfolder'] = outputfolder
    kwargs['inputfolder'] = inputfolder
    run_parallel(process=__resize_batch_helper, process_name='Resize', **kwargs)


def add_noise_batch(inputfolder, outputfolder, noise_kind, snr, test_snr_combinations=False, **kwargs):
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
    noise_kind : string, sequence of strings or None
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

    kind = np.array([noise_kind]).flatten()
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


def __generate_cells_batch_helper(item, outputfolder, input_voxel_size, **kwargs_to_ignore):
    filename, params = item
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    cell = Cell(input_voxel_size=input_voxel_size, **dict(params))
    cell.save(outputfolder + filename)
    cell.metadata['CellID'] = filename[:-4].split('_')[-1]
    cell.metadata.save(outputfolder + filename[:-4] + '.csv')


def __generate_stacks_batch_helper(item, outputfolder, input_voxel_size,
                                   stack_size_microns, **kwargs_to_ignore):
    filename, params = item
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    stack = Stack(input_voxel_size=input_voxel_size, stack_size=stack_size_microns, cell_params=params)
    stack.save(outputfolder + filename[:-4] + '.tif')
    stack.metadata['StackID'] = filename[:-4].split('_')[-1]
    stack.metadata.save(outputfolder + filename[:-4] + '.csv')


def __generate_psfs_batch_helper(item, outputfolder, input_voxel_size, **kwargs_to_ignore):
    metadata = Metadata()
    metadata.set_voxel_size(input_voxel_size)
    sigma, aspect_ratio = item
    sigmaz = sigma * aspect_ratio
    sigmax = sigma / metadata['Voxel size x']
    sigmaz = sigmaz / metadata['Voxel size z']
    psf = PSF(sigma=sigmax, aspect_ratio=sigmaz / sigmax)
    psf.save(outputfolder + 'psf_sigma_' + str(sigma) + '_aspect_ratio_' + str(aspect_ratio) + '.tif',
             normalize_output=True)
    metadata['PSF sigma xy um'] = sigma
    metadata['PSF aspect ratio'] = aspect_ratio
    metadata['isPSF'] = True
    metadata.save(outputfolder + 'psf_sigma_' + str(sigma) + '_aspect_ratio_' + str(aspect_ratio) + '.csv')


def __convolve_batch_helper(item, inputfolder, psffolder, outputfolder, **kwargs_to_ignore):
    inputfile, psffile = item
    stack = Stack(filename=inputfolder + inputfile)
    psf = PSF(filename=psffolder + psffile)
    stack.convolve(psf)
    stack.save(outputfolder + psffile[:-4] + '/' + inputfile)
    stack.metadata.save(outputfolder + psffile[:-4] + '/' + inputfile[:-4] + '.csv')
    psf.save(outputfolder + psffile)
    psf.metadata.save(outputfolder + psffile[:-4] + '.csv')


def __resize_batch_helper(item, inputfolder, outputfolder, order=1, append_resolution_to_filename=True,
                          **kwargs_to_ignore):
    inputfile, voxel_size = item
    metadata = Metadata()
    metadata.set_voxel_size(voxel_size)
    stack = Stack(filename=inputfolder + inputfile)
    zoom = np.array(stack.metadata['Voxel size arr']) / np.array(metadata['Voxel size arr'])
    stack.resize(zoom=zoom, order=order)
    stack.metadata.set_voxel_size(metadata['Voxel size arr'])
    path = os.path.dirname(inputfile)
    strres = str(metadata['Voxel size']).replace('  ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ', '_')
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
    if 'isPSF' not in stack.metadata.index or str(stack.metadata['isPSF']) == 'False':
        stack.add_noise(kind=kind, snr=snr)

    if path == '':
        outputname = outputfolder + inputfile
    else:
        outputname = outputfolder + path + '_noise'
        snr = np.array([snr]).flatten()
        kind = np.array([kind]).flatten()
        if len(snr) == 1:
            snr = np.array([snr[0]] * len(kind))
        for i, k in enumerate(kind):
            outputname = outputname + '_' + k + '_snr=' + str(snr[i])
        outputname = outputname + '/' + inputfile.split('/')[-1]
    stack.save(outputname)
    stack.metadata.save(outputname[:-4] + '.csv')
