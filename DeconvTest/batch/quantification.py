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


def accuracy_batch(inputfolder, outputfolder, combine_stat=True, **kwargs):
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

    if not os.path.exists(kwargs['reffolder']):
        kwargs['items'] = filelib.list_subfolders(inputfolder)
        warnings.warn('Reference directory ' + kwargs['reffolder'] + ' does not exist!')
    kwargs['inputfolder'] = inputfolder
    kwargs['outputfolder'] = outputfolder
    run_parallel(process=__compute_accuracy_measures_batch_helper,
                 process_name='Compute accuracy measures', **kwargs)

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


def __compute_accuracy_measures_batch_helper(item, inputfolder, reffolder, outputfolder, **kwargs_to_ignore):
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
            refstack = Stack(filename=reffolder + item)
        elif os.path.exists(reffolder + name):
            refstack = Stack(filename=reffolder + name)
        elif os.path.exists(reffolder + base + '/' + name):
            refstack = Stack(filename=reffolder + base + '/' + name)
        elif os.path.exists(reffolder + name.split('_voxel_size')[0] + '.tif'):
            refstack = Stack(filename=reffolder + name.split('_voxel_size')[0] + '.tif')
        else:
            raise ValueError('No ground truth found for cell ' + item + '!')

        input_voxel_size = stack.metadata['Voxel size arr']
        zoom = np.array(stack.metadata['Voxel size arr']) / np.array(refstack.metadata['Voxel size arr'])
        stack.resize(zoom=zoom)
        stats = stack.compute_accuracy_measures(refstack)

        stack.metadata.set_voxel_size(input_voxel_size)
        for c in stack.metadata.index:
            try:
                stats[c] = stack.metadata[c]
            except ValueError:
                stats[c] = str(stack.metadata[c])
        stats['Name'] = item

        filelib.make_folders([os.path.dirname(outputfolder + item)])
        stats.to_csv(outputfolder + item[:-4] + '.csv', sep='\t')

