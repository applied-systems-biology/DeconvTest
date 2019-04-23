from __future__ import division

import sys
import re
import numpy as np
import pandas as pd
import os
import time
from helper_lib import filelib

from DeconvTest.batch import simulation as sim
from DeconvTest import batch
from DeconvTest.modules.deconvolution import save_fiji_version

import mkl
mkl.set_num_threads(1)


def convert_args(**kwargs):
    for c in ['max_threads', 'number_of_stacks', 'thr']:
        if c in kwargs:
            if type(kwargs[c]) is str:
                if kwargs[c] == 'None':
                    kwargs[c] = None
                else:
                    kwargs[c] = float(kwargs[c])

    p = re.compile('\d*\.*\d+')  # create a template for extracting nonnegative float numbers
    if type(kwargs['input_voxel_size']) is str:
        kwargs['input_voxel_size'] = float(p.findall(kwargs['input_voxel_size'])[0])

    if type(kwargs['number_of_cells']) is str:
        nums = p.findall(kwargs['number_of_cells'])
        if len(nums) > 1:
            kwargs['number_of_cells'] = np.float_(nums)
        else:
            kwargs['number_of_cells'] = float(nums[0])

    for c in ['size_mean_and_std', 'spikiness_range', 'spike_size_range',
              'spike_smoothness_range', 'stack_size_microns',
              'psf_sigmas', 'psf_aspect_ratios', 'snr',
              'deconvolution_lab_rltv_iterations', 'iterative_deconvolve_3d_low',
              'deconvolution_lab_rif_regularization_lambda', 'deconvolution_lab_rltv_regularization_lambda',
              'iterative_deconvolve_3d_terminate', 'iterative_deconvolve_3d_wiener']:
        if c in kwargs:
            if type(kwargs[c]) is str:
                nums = np.float_(p.findall(kwargs[c]))
                if len(kwargs[c].split('None')) > 1:
                    nums = np.concatenate((nums, np.array([None])))
                kwargs[c] = nums

    for c in ['print_progress', 'equal_dimensions', 'preprocess', 'relative_thr',
              'postprocess', 'log_computing_time']:
        if str(kwargs[c]).upper() == 'FALSE':
            kwargs[c] = False
        else:
            kwargs[c] = True

    if type(kwargs['voxel_sizes_for_resizing']) is str:
        strvox = kwargs['voxel_sizes_for_resizing']
        strvox = strvox.replace(' ', '')
        p1 = re.compile('\[([A-Za-z0-9_,.]+)\]')
        arrays = p1.findall(strvox)
        voxel_sizes = []
        for arr in arrays:
            voxel_sizes.append(np.float_(p.findall(arr)))
            strvox = strvox.replace('['+arr+']', '')
        nums = np.float_(p.findall(strvox))
        for n in nums:
            voxel_sizes.append(round(n, 7))
        kwargs['voxel_sizes_for_resizing'] = voxel_sizes
    for c in ['deconvolution_algorithm', 'noise_kind']:
        if c in kwargs:
            if type(kwargs[c]) is str:
                stralg = '[' + str(kwargs[c]) + ']'
                p1 = re.compile('([A-Za-z0-9_]+)')
                kwargs[c] = p1.findall(stralg)

    for c in ['iterative_deconvolve_3d_detect', 'iterative_deconvolve_3d_perform',
              'iterative_deconvolve_3d_normalize']:
        if c in kwargs:
            p1 = re.compile('([A-Za-z]+)')
            parts = p1.findall(str(kwargs[c]))
            arrays = []
            for part in parts:
                if part.upper() == 'TRUE':
                    arrays.append(True)
                else:
                    arrays.append(False)
            kwargs[c] = arrays
    return kwargs


def run_simulation(**kwargs):
    steps = kwargs['simulation_steps']
    simulation_folder = kwargs.get('simulation_folder')
    if not simulation_folder.endswith('/'):
        simulation_folder += '/'
    filelib.make_folders([simulation_folder])

    kwargs = convert_args(**kwargs)
    kwargs['Time of the simulation start'] = time.ctime()
    pd.Series(kwargs).to_csv(simulation_folder + 'simulation_parameters.csv', sep='\t')
    save_fiji_version(simulation_folder)
    kwargs['logfolder'] = simulation_folder + kwargs['logfolder']

    for step in steps:
        if step == 'generate_cells':
            inputfolder = simulation_folder + kwargs['cell_parameter_filename']
            if not os.path.exists(inputfolder):
                print 'Generating new cell parameters'
                sim.generate_cell_parameters(outputfile=inputfolder, **kwargs)
            batch.generate_cells_batch(params_file=inputfolder,
                                       outputfolder=simulation_folder + kwargs['inputfolder'],
                                       **kwargs)
            kwargs['inputfolder'] = simulation_folder + kwargs['inputfolder']
            kwargs['reffolder'] = kwargs['inputfolder']
        elif step == 'generate_psfs':
            kwargs['psffolder'] = simulation_folder + kwargs['psffolder']
            batch.generate_psfs_batch(outputfolder=kwargs['psffolder'], **kwargs)
        else:
            kwargs['outputfolder'] = simulation_folder + kwargs[step + '_results_folder']
            getattr(batch, step + '_batch')(**kwargs)
            kwargs['inputfolder'] = kwargs['outputfolder']

    batch.combine_log(inputfolder=kwargs['logfolder'])


########################################

default_parameters = dict({'simulation_folder': 'test_simulation',
                           'simulation_steps': ['generate_cells', 'generate_psfs', 'convolve', 'resize', 'add_noise',
                                                'deconvolve', 'binary_accuracy'],
                           'cell_parameter_filename': 'cell_parameters.csv',
                           'inputfolder': 'input',
                           'psffolder': 'psf',
                           'convolve_results_folder': 'convolved',
                           'resize_results_folder': 'resized',
                           'add_noise_results_folder': 'noise',
                           'deconvolve_results_folder': 'deconvolved',
                           'binary_accuracy_results_folder': 'binary_accuracy_measures',
                           'logfolder': 'timelog',
                           'max_threads': 4,
                           'print_progress': True,
                           'number_of_stacks': None,
                           'number_of_cells': 2,
                           'input_cell_kind': 'ellipsoid',
                           'size_mean_and_std': (10, 2),
                           'equal_dimensions': False,
                           'input_voxel_size': 0.3,
                           'stack_size_microns': [10, 100, 100],
                           'psf_sigmas': [0.1, 0.5],
                           'psf_aspect_ratios': [3],
                           'voxel_sizes_for_resizing': [[1, 0.5, 0.5]],
                           'noise_kind': ['poisson'],
                           'snr': [None, 5],
                           'test_snr_combinations': False,
                           'deconvolution_algorithm': ['deconvolution_lab_rif', 'deconvolution_lab_rltv'],
                           'deconvolution_lab_rif_regularization_lambda': [0.001, 001],
                           'deconvolution_lab_rltv_regularization_lambda': 0.001,
                           'deconvolution_lab_rltv_iterations': [2, 3],
                           'log_computing_time': True,
                           'preprocess': False,
                           'thr': None,
                           'relative_thr': False,
                           'postprocess': False
                            })


if __name__ == '__main__':

    args = sys.argv[1:]
    step = None
    if len(args) > 0:
        params = dict(pd.read_csv(args[0], sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze())
        for c in default_parameters:
            if c not in params:
                params[c] = default_parameters[c]
    else:
        params = default_parameters
    run_simulation(**params)


