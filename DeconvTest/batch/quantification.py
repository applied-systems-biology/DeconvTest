"""
Module containing functions for evaluating deconvolution accuracy in a batch mode. 
Includes functions for stack segmentation, measuring cell dimensions and comparing to ground truth.
"""
from __future__ import division

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

from DeconvTest import Stack
from DeconvTest import Metadata
from helper_lib.parallel import run_parallel
from helper_lib import filelib


def segment_batch(inputfolder, outputfolder, preprocess=False, thr=None, relative_thr=False,
                  postprocess=False, max_threads=8, print_progress=True,
                  log_computing_time=False, logfolder=None):
    """
    Segments all cell images in a given input directory in a parallel mode and saves them in a given output directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to segment.
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
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    kwargs = {'items': inputfiles, 'inputfolder': inputfolder, 'outputfolder': outputfolder,
              'preprocess': preprocess, 'thr': thr, 'relative_thr': relative_thr,
              'postprocess': postprocess, 'max_threads': max_threads, 'print_progress': print_progress,
              'log_computing_time': log_computing_time, 'logfolder': logfolder}
    run_parallel(process=__segment_batch_helper, process_name='Segment', **kwargs)


def compare_to_ground_truth_batch(inputfolder, reffolder, outputfolder, max_threads=8, print_progress=True,
                                  combine_stat=True):
    """
    Compares all images in a given input directory to corresponding ground truth images in a give reference directory.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to compare.
    reffolder : str
        Reference dirctory with ground truth cell images.
    outputfolder : str
        Output directory to save the computed accuracy values.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.
    combine_stat : bool, optional
        If True, the statistics for all cells will be combined into one csv file.
        Default is True.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if not reffolder.endswith('/'):
        reffolder += '/'
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    kwargs = {'items': inputfiles, 'inputfolder': inputfolder, 'outputfolder': outputfolder,
              'reffolder': reffolder, 'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__compare_to_ground_truth_batch_helper, process_name='Compare to ground truth', **kwargs)
    if os.path.exists(outputfolder) and combine_stat is True:
        filelib.combine_statistics(outputfolder)


def measure_dimensions_batch(inputfolder, outputfolder, max_threads=8, print_progress=True):
    """
    Measures dimensions of all cell images in a given input directory in a parallel mode.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to measure dimensions.
    outputfolder : str
        Output directory to save the measured dimension values.
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
    kwargs = {'items': inputfiles, 'inputfolder': inputfolder, 'outputfolder': outputfolder,
              'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__measure_dimensions_batch_helper, process_name='Measure dimensions', **kwargs)
    if os.path.exists(outputfolder):
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
        data = pd.DataFrame()

        for sf in subfolders:
            curdata = pd.read_csv(inputfolder + sf, sep='\t', index_col=0)
            data = pd.concat([data, curdata], ignore_index=True, sort=False)

        if len(data) > 0:
            data.to_csv(inputfolder[:-1] + '.csv', sep='\t')


def compute_dimension_errors(inputfile, default_resolution, outputfile=None):
    """    
    Computes dimension errors between dimensions of input cells and reference cells

    Parameters
    ----------
    inputfile : str
        Path to the csv file with measured dimensions of the input and reference cells.
    default_resolution : scalar or sequence of scalars
        Default voxel size in z, y and x used to compute dimensions of the input cells.
        If one value is provided, the voxel size is assume to be equal along all axes.
    outputfile : str, optional
        Path to a file where to store the computed dimension errors.
        If None, the output path is given by `inputfile`[:-4] + "_errors.csv".
        Default is None.
    """
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)  # read dimensions
    if outputfile is None:
        outputfile = inputfile[:-4] + '_errors.csv'

    if 'Kind' not in stat.columns or 'Cell ID' not in stat.columns or 'Voxel size x' not in stat.columns \
            or 'size_x' not in stat.columns:
        nstat = pd.DataFrame()
        for i in range(len(stat)):
            metadata = Metadata(resolution=default_resolution, string=stat.iloc[i]['Name'])
            if metadata['resolution'][0] == metadata['resolution'][1] == metadata['resolution'][2]:
                metadata['Resolution'] = str(metadata['resolution'][0])
            else:
                metadata['Resolution'] = str(metadata['resolution'])

            nstat = nstat.append(metadata, ignore_index=True)
        for c in nstat.columns:
            if c not in stat.columns:
                stat[c] = nstat[c]

    # recompute the sizes into micrometers
    for i in ['x', 'y', 'z']:
        stat['size_' + i] = stat['size_' + i] * stat['Voxel size ' + i]

    # stat.to_csv(inputfile[:-4] + '_metadata.csv', sep='\t')

    # split statistics into reference (input) and test (segmented/convolved)
    stat_input = stat[stat['Kind'] == 'Input'].reset_index(drop=True)
    stat_conv = stat[stat['Kind'] == 'Convolved'].reset_index(drop=True)

    # find and combine corresponding cells from test and reference
    groups = stat_conv.groupby('CellID').groups
    for id in stat_conv['CellID'].unique():
        cur_input = stat_input[(stat_input['CellID'] == id)]
        if len(cur_input) > 0:
            cur_input = cur_input.iloc[0]
        for i in ['x', 'y', 'z']:
            size = cur_input['size_' + i]
            stat_conv.at[stat_conv[stat_conv['CellID'] == id].index, 'size_' + i + ' input'] = size

    # compute the differences between corresponding cells
    for i in ['x', 'y', 'z']:
        stat_conv['d' + i] = (stat_conv['size_' + i] - stat_conv['size_' + i + ' input']) / 2.
        stat_conv['d' + i + '_norm'] = stat_conv['d' + i] / stat_conv['size_' + i + ' input']
        stat_conv['d' + i + '_abs'] = np.abs(stat_conv['d' + i])

    stat_conv.to_csv(outputfile, sep='\t')


def extract_metadata(inputfile, default_resolution, outputfile=None):
    """    
    Extract metadata for each row of the csv file using the field `Name`.

    Parameters
    inputfile : str
        Path to the csv file to extract metadata from.
    default_resolution : scalar or sequence of scalars
        Default voxel size in z, y and x that was used to generate the input cells.
        If one value is provided, the voxel size is assume to be equal along all axes.
    outputfile : str, optional
        Path to a file where to store the output.
        If None, the input file will be overwritten.
        Default is None.
    """
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)

    nstat = pd.DataFrame()
    for i in range(len(stat)):
        metadata = Metadata(resolution=default_resolution, string=stat.iloc[i]['Name'])
        if metadata['resolution'][0] == metadata['resolution'][1] == metadata['resolution'][2]:
            metadata['Resolution'] = str(metadata['resolution'][0])
        else:
            metadata['Resolution'] = str(metadata['resolution'])
        nstat = nstat.append(metadata, ignore_index=True)
    for c in nstat.columns:
        if c not in stat.columns:
            stat[c] = nstat[c]

    if outputfile is None:
        outputfile = inputfile
    stat.to_csv(outputfile, sep='\t')


####################################
# private helper functions


def __segment_batch_helper(item, inputfolder, outputfolder, preprocess, thr, relative_thr,
                           postprocess, log_computing_time, logfolder):
    name = item.split('/')[-1]
    if len(name.split('psf')) == 1:
        stack = Stack(filename=inputfolder + item)
        start = time.time()
        stack.segment(preprocess=preprocess, thr=thr, relative_thr=relative_thr, postprocess=postprocess, label=True)
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
            t.to_csv(logfolder + item[:-4].replace('/', '_') + '.csv', sep='\t')


def __compare_to_ground_truth_batch_helper(item, inputfolder, reffolder, outputfolder):
    parts = item.split('/')
    name = parts[-1]
    if len(parts) > 1:
        base = parts[-2]
    else:
        base = ''
    if len(name.split('psf')) == 1:
        stack = Stack(filename=inputfolder + item, is_segmented=True)
        if os.path.exists(reffolder + item):
            refstack = Stack(filename=reffolder + item, is_segmented=True)
        elif os.path.exists(reffolder + name):
            refstack =Stack(filename=reffolder + name, is_segmented=True)
        elif os.path.exists(reffolder + base + '/' + name):
            refstack =Stack(filename=reffolder + base + '/' + name, is_segmented=True)
        elif os.path.exists(reffolder + name.split('_voxel_size')[0] + '.tif'):
            refstack =Stack(filename=reffolder + name.split('_voxel_size')[0] + '.tif', is_segmented=True)
        else:
            raise ValueError('No ground truth found for cell ' + item + '!')

        zoom = np.array(stack.metadata['resolution']) / np.array(refstack.metadata['resolution'])
        stack.resize(zoom=zoom)
        stack.segment()
        stats = stack.compare_to_ground_truth(refstack)
        stats['Name'] = item
        filelib.make_folders([os.path.dirname(outputfolder + item)])
        stats.to_csv(outputfolder + item[:-4] + '.csv', sep='\t')


def __measure_dimensions_batch_helper(item, inputfolder, outputfolder):
    name = item.split('/')[-1]
    if len(name.split('psf')) == 1:
        stack = Stack(filename=inputfolder + item, is_segmented=True)
        stack.split_to_cells()
        dimensions = stack.dimensions()
        stats = pd.DataFrame()
        dimension_xyz = ['z', 'y', 'x']
        for i in range(len(dimensions[0])):
            stats['size_' + dimension_xyz[i]] = dimensions[:, i]

        stats['Name'] = item
        filelib.make_folders([os.path.dirname(outputfolder + item)])
        stats.to_csv(outputfolder + item[:-4] + '.csv', sep='\t')

