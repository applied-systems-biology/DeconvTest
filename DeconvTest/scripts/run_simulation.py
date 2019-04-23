from __future__ import division

import sys
import re
import numpy as np
import pandas as pd
import os
import time
from helper_lib import filelib

from DeconvTest.batch import simulation as sim
from DeconvTest.batch import quantification as quant
from DeconvTest.batch import deconvolution as dec
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
                stralg = str(kwargs[c])
                p1 = re.compile('\'([A-Za-z0-9_,.]+)\'')
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
    outputfolder = simulation_folder + kwargs['inputfolder']

    kwargs = convert_args(**kwargs)
    kwargs['Time of the simulation start'] = time.ctime()
    pd.Series(kwargs).to_csv(simulation_folder + 'simulation_parameters.csv', sep='\t')
    save_fiji_version(simulation_folder)

    for step in steps:
        print step
        if step == 'generate_cells':
            inputfolder = simulation_folder + kwargs['cell_parameter_filename']
            if not os.path.exists(inputfolder):
                print 'Generating new cell parameters'
                sim.generate_cell_parameters(outputfile=inputfolder, **kwargs)
            sim.generate_cells_batch(params_file=inputfolder, outputfolder=outputfolder, **kwargs)


    #
    #
    # if step is None or step == 1:
    #     number_of_stacks = params.get('number_of_stacks')
    #     if number_of_stacks is None:
    #         sim.generate_cell_parameters(outputfile=cell_parameter_filename,
    #                                      number_of_cells=params.get('number_of_cells'),
    #                                      size_mean_and_std=params.get('size_mean_and_std'),
    #                                      equal_dimensions=params.get('equal_dimensions'),
    #                                      spikiness_range=params.get('spikiness_range'),
    #                                      spike_size_range=params.get('spike_size_range'),
    #                                      spike_smoothness_range=params.get('spike_smoothness_range'))
    #         sim.generate_cells_batch(params_file=cell_parameter_filename, outputfolder=inputfolder,
    #                                  resolution=params.get('input_voxel_size'),
    #                                  max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
    #     else:
    #         number_of_stacks = int(float(number_of_stacks))
    #         sim.generate_stack_parameters(outputfolder=cell_parameter_folder, number_of_stacks=number_of_stacks,
    #                                       number_of_cells=params.get('number_of_cells'),
    #                                       size_mean_and_std=params.get('size_mean_and_std'),
    #                                       equal_dimensions=params.get('equal_dimensions'),
    #                                       spikiness_range=params.get('spikiness_range'),
    #                                       spike_size_range=params.get('spike_size_range'),
    #                                       spike_smoothness_range=params.get('spike_smoothness_range'))
    #         sim.generate_stacks_batch(params_folder=cell_parameter_folder, outputfolder=inputfolder,
    #                                   resolution=params.get('input_voxel_size'),
    #                                   stack_size_microns=params.get('stack_size_microns'),
    #                                   max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
    #
    #     sim.generate_psfs_batch(outputfolder=psffolder, sigmas=params.get('psf_sigmas'),
    #                             elongations=params.get('psf_elongations'), resolution=params.get('input_voxel_size'),
    #                             max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
    #     sim.convolve_batch(inputfolder=inputfolder, psffolder=psffolder, outputfolder=convolution_folder,
    #                        max_threads=int(params.get('max_threads')/3), print_progress=params.get('print_progress'))
    #     sim.resize_batch(inputfolder=convolution_folder, outputfolder=resizing_folder,
    #                      resolutions=params.get('voxel_sizes_for_resizing'),
    #                      max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
    #     sim.add_noise_batch(inputfolder=resizing_folder, outputfolder=noise_folder,
    #                         gaussian_snrs=params.get('gaussian_snrs'), poisson_snrs=params.get('poisson_snrs'),
    #                         max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
    #
    # if step is None or step == 2:
    #     dec.deconvolve_batch(inputfolder=noise_folder, outputfolder=deconvolution_folder,
    #                          algorithm=params.get('algorithm'), rif_lambda=params.get('rif_lambda'),
    #                          rltv_lambda=params.get('rltv_lambda'), iterations=params.get('iterations'),
    #                          normalize=params.get('normalize'), perform=params.get('perform'),
    #                          detect=params.get('detect'), wiener=params.get('wiener'), low=params.get('low'),
    #                          terminate=params.get('terminate'), log_computing_time=True,
    #                          logfolder=log_folder, print_progress=params.get('print_progress'))
    #
    # if step is None or step == 3:
    #     quant.segment_batch(inputfolder=deconvolution_folder, outputfolder=segmentation_folder,
    #                         preprocess=params.get('preprocess'), thr=params.get('thr'),
    #                         relative_thr=params.get('relative_thr'), postprocess=params.get('postprocess'),
    #                         max_threads=params.get('max_threads'), print_progress=params.get('print_progress'),
    #                         log_computing_time=False, logfolder=log_folder)
    #     quant.compare_to_ground_truth_batch(inputfolder=segmentation_folder, reffolder=inputfolder,
    #                                         outputfolder=accuracy_folder, max_threads=params.get('max_threads'),
    #                                         print_progress=params.get('print_progress'))
    #     quant.extract_metadata(inputfile=accuracy_folder[:-1] + '.csv', default_resolution=params.get('input_voxel_size'))
    #     quant.combine_log(inputfolder=log_folder)
    #     quant.extract_metadata(inputfile=log_folder[:-1] + '.csv', default_resolution=params.get('input_voxel_size'))


########################################

default_parameters = dict({'simulation_folder': 'test_simulation',
                           'simulation_steps': ['generate_cells', 'generate_psfs', 'convolve', 'resize', 'add_noise',
                                                'deconvolve', 'compute_binary_accuracy_measures'],
                           'cell_parameter_filename': 'cell_parameters.csv',
                           'inputfolder': 'input',
                           'psffolder': 'psf',
                           'convole_results_folder': 'convolved',
                           'resize_results_folder': 'resized',
                           'add_noise_results_folder': 'noise',
                           'deconvolve_results_folder': 'deconvolved',
                           'folder_for_binary_accuracy_measures': 'binary_accuracy_measrues',
                           'log_folder': 'timelog',
                           'max_threads': 4,
                           'print_progress': True,
                           'number_of_stacks': 2,
                           'number_of_cells': 2,
                           'input_cell_kind': 'ellipsoid',
                           'size_mean_and_std': (10, 2),
                           'equal_dimensions': False,
                           'input_voxel_size': 0.3,
                           'stack_size_microns': [10, 100, 100],
                           'psf_sigmas': [0.1, 0.5],
                           'psf_aspect_ratios': [3],
                           'voxel_sizes_for_resizing': [[1, 0.5, 0.5]],
                           'noise_kind': 'poisson',
                           'snr': [None, 5],
                           'deconvolution_algorithm': ['deconvolution_lab_rif', 'deconvolution_lab_rltv'],
                           'deconvolution_lab_rif_regularization_lambda': [0.001, 001],
                           'deconvolution_lab_rltv_regularization_lambda': 0.001,
                           'deconvolution_lab_rltv_iterations': [2, 5],
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


