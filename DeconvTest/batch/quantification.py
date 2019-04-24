"""
Module containing functions for evaluating deconvolution accuracy in a batch mode. 
Includes functions for stack segmentation, measuring cell dimensions and comparing to ground truth.
"""
from __future__ import division

import os
import time
import warnings
import numpy as np
import pandas as pd


from DeconvTest import Stack
from helper_lib.parallel import run_parallel
from helper_lib import filelib


def segment_batch(inputfolder, **kwargs):
    """
    Segments all cell images in a given input directory in a parallel mode and saves them in a given output directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to segment.

    Keyword arguments
    -----------------
    outputfolder : str
        Output directory to save the segmented images.
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
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.
    log_computing_time : bool, optional
        If True, computing time spent on segmentation will be recorded and stored in a given folder.
        Default is False.
    logfolder : str, optional
        Directory to store computing time when `log_computing_time` is set to True.
        If None, the logfolder will be set to `outputfolder` + "../log/".
        Default is None.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    kwargs['inputfolder'] = inputfolder
    kwargs['items'] = filelib.list_subfolders(inputfolder)
    run_parallel(process=__segment_batch_helper, process_name='Segment', **kwargs)


def binary_accuracy_batch(inputfolder, outputfolder, combine_stat=True, **kwargs):
    """
    Compares all images in a given input directory to corresponding ground truth images in a give reference directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to compare.
    outputfolder : str
        Output directory to save the computed accuracy values.
    combine_stat : bool, optional
        If True, the statistics for all cells will be combined into one csv file.
        Default is True.

    Keyword arguments
    -----------------
    reffolder : str
        Reference dirctory with ground truth cell images.
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
    if os.path.exists(inputfolder):
        kwargs['items'] = filelib.list_subfolders(inputfolder)
    else:
        kwargs['items'] = []
        warnings.warn('Input directory ' + inputfolder +
                      ' does not exist!')

    kwargs['inputfolder'] = inputfolder
    kwargs['outputfolder'] = outputfolder
    run_parallel(process=__compute_binary_accuracy_measures_batch_helper,
                 process_name='Compute binary accuracy measures', **kwargs)

    if os.path.exists(outputfolder) and combine_stat is True:
        filelib.combine_statistics(outputfolder)


def combine_log(inputfolder):
    """
    Combines the files with computing time that are stored in a given directory.
    
    Parameters
    ----------
    inputfolder : str
        Directory with the computing time logs.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if os.path.exists(inputfolder):
        subfolders = filelib.list_subfolders(inputfolder, extensions=['csv'])
        if len(subfolders) > 0:
            array = []
            for i, sf in enumerate(subfolders):
                data = pd.read_csv(inputfolder + sf, sep='\t', index_col=0)
                array.append(data)
            data = pd.concat(array, ignore_index=True, sort=True)
            data.to_csv(inputfolder[:-1] + '.csv', sep='\t')


####################################
# private helper functions


def __segment_batch_helper(item, inputfolder, outputfolder, log_computing_time=False, logfolder=None, **kwargs):
    name = item.split('/')[-1]
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    if len(name.split('psf')) == 1:  # only analyze files that don't have 'psf' in their file names
        stack = Stack(filename=inputfolder + item)
        start = time.time()
        stack.segment(**kwargs)
        elapsed_time = time.time() - start
        stack.save(outputfolder + item)
        stack.metadata.save(outputfolder + item[:-4] + '.csv')
        if log_computing_time is True:
            if logfolder is None:
                logfolder = outputfolder + '../log/'
            else:
                if not logfolder.endswith('/'):
                    logfolder += '/'

            filelib.make_folders([logfolder])
            t = pd.DataFrame({'Step': ['Segmentation'],
                              'Computational time': [elapsed_time],
                              'Name': item})
            for c in stack.metadata.index:
                try:
                    t[c] = stack.metadata[c]
                except ValueError:
                    t[c] = str(stack.metadata[c])
            t.to_csv(logfolder + item[:-4].replace('/', '_') + '.csv', sep='\t')


def __compute_binary_accuracy_measures_batch_helper(item, inputfolder, reffolder, outputfolder,
                                                    log_computing_time=False, logfolder=None, **segmentation_kwargs):
    if not reffolder.endswith('/'):
        reffolder += '/'
    parts = item.split('/')
    name = parts[-1]
    if len(parts) > 1:
        base = parts[-2]
    else:
        base = ''
    stack = Stack(filename=inputfolder + item)
    if 'isPSF' not in stack.metadata.index or str(stack.metadata['isPSF']) == 'False':
        if os.path.exists(reffolder + item):
            refstack = Stack(filename=reffolder + item, is_segmented=True)
        elif os.path.exists(reffolder + name):
            refstack = Stack(filename=reffolder + name, is_segmented=True)
        elif os.path.exists(reffolder + base + '/' + name):
            refstack = Stack(filename=reffolder + base + '/' + name, is_segmented=True)
        elif os.path.exists(reffolder + name.split('_voxel_size')[0] + '.tif'):
            refstack = Stack(filename=reffolder + name.split('_voxel_size')[0] + '.tif', is_segmented=True)
        else:
            raise ValueError('No ground truth found for cell ' + item + '!')

        start = time.time()
        input_voxel_size = stack.metadata['Voxel size arr']
        zoom = np.array(stack.metadata['Voxel size arr']) / np.array(refstack.metadata['Voxel size arr'])
        stack.resize(zoom=zoom)
        stack.segment(**segmentation_kwargs)
        stats = stack.compute_binary_accuracy_measures(refstack)

        stack.metadata.set_voxel_size(input_voxel_size)
        for c in stack.metadata.index:
            try:
                stats[c] = stack.metadata[c]
            except ValueError:
                stats[c] = str(stack.metadata[c])
        stats['Name'] = item

        filelib.make_folders([os.path.dirname(outputfolder + item)])
        stats.to_csv(outputfolder + item[:-4] + '.csv', sep='\t')
        elapsed_time = time.time() - start
        if log_computing_time is True:
            if logfolder is None:
                logfolder = outputfolder + '../log/'
            else:
                if not logfolder.endswith('/'):
                    logfolder += '/'

            filelib.make_folders([logfolder])
            t = pd.DataFrame({'Step': ['Computing binary accuracy measures'],
                              'Computational time': [elapsed_time],
                              'Name': item})
            for c in stack.metadata.index:
                try:
                    t[c] = stack.metadata[c]
                except ValueError:
                    t[c] = str(stack.metadata[c])
            t.to_csv(logfolder + item[:-4].replace('/', '_') + '.csv', sep='\t')


