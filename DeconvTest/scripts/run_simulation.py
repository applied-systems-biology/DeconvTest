from __future__ import division

import sys
import re
import numpy as np
import pandas as pd
import time
from helper_lib import filelib

from DeconvTest.batch import simulation as sim
from DeconvTest.batch import quantification as quant
from DeconvTest.batch import deconvolution as dec
from DeconvTest.deconvolve.fiji import save_fiji_version

import mkl
mkl.set_num_threads(1)


def convert_args(**kwargs):
    for c in ['max_threads', 'number_of_stacks', 'thr']:
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
              'psf_sigmas', 'psf_elongations', 'poisson_snrs', 'gaussian_snrs',
              'iterations', 'low', 'rif_lambda', 'rltv_lambda', 'terminate', 'wiener']:
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

    if type(kwargs['algorithm']) is str:
        stralg = str(kwargs['algorithm'])
        p1 = re.compile('\'([A-Za-z0-9_,.]+)\'')
        kwargs['algorithm'] = p1.findall(stralg)

    for c in ['detect', 'perform', 'normalize']:
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


def run_simulation(step=None, **params):
    simulation_folder = params.get('simulation_folder')
    filelib.make_folders([simulation_folder])
    if not simulation_folder.endswith('/'):
        simulation_folder += '/'

    cell_parameter_filename = simulation_folder + params.get('cell_parameter_filename')
    if not cell_parameter_filename.endswith('.csv'):
        cell_parameter_filename += '.csv'

    cell_parameter_folder = simulation_folder + params.get('cell_parameter_folder')
    if not cell_parameter_folder.endswith('/'):
        cell_parameter_folder += '/'

    inputfolder = simulation_folder + params.get('inputfolder')
    if not inputfolder.endswith('/'):
        inputfolder += '/'

    psffolder = simulation_folder + params.get('psffolder')
    if not psffolder.endswith('/'):
        psffolder += '/'

    convolution_folder = simulation_folder + params.get('convolution_results_folder')
    if not convolution_folder.endswith('/'):
        convolution_folder += '/'

    resizing_folder = simulation_folder + params.get('resizing_results_folder')
    if not resizing_folder.endswith('/'):
        resizing_folder += '/'

    noise_folder = simulation_folder + params.get('noise_results_folder')
    if not noise_folder.endswith('/'):
        noise_folder += '/'

    deconvolution_folder = simulation_folder + params.get('deconvolution_results_folder')
    if not deconvolution_folder.endswith('/'):
        deconvolution_folder += '/'

    segmentation_folder = simulation_folder + params.get('segmentation_results_folder')
    if not segmentation_folder.endswith('/'):
        segmentation_folder += '/'

    log_folder = simulation_folder + params.get('log_folder')
    if not log_folder.endswith('/'):
        log_folder += '/'

    accuracy_folder = simulation_folder + params.get('accuracy_results_folder')
    if not accuracy_folder.endswith('/'):
        accuracy_folder += '/'

    dimensions_folder = simulation_folder + params.get('dimensions_folder')
    if not dimensions_folder.endswith('/'):
        dimensions_folder += '/'

    params = convert_args(**params)
    params['Time of the simulation start'] = time.ctime()
    pd.Series(params).to_csv(simulation_folder + 'simulation_parameters.csv', sep='\t')
    save_fiji_version(simulation_folder)

    if step is None or step == 1:
        number_of_stacks = params.get('number_of_stacks')
        if number_of_stacks is None:
            sim.generate_cell_parameters(outputfile=cell_parameter_filename,
                                         number_of_cells=params.get('number_of_cells'),
                                         size_mean_and_std=params.get('size_mean_and_std'),
                                         equal_dimensions=params.get('equal_dimensions'),
                                         spikiness_range=params.get('spikiness_range'),
                                         spike_size_range=params.get('spike_size_range'),
                                         spike_smoothness_range=params.get('spike_smoothness_range'))
            sim.generate_cells_batch(params_file=cell_parameter_filename, outputfolder=inputfolder,
                                     resolution=params.get('input_voxel_size'),
                                     max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
        else:
            number_of_stacks = int(float(number_of_stacks))
            sim.generate_stack_parameters(outputfolder=cell_parameter_folder, number_of_stacks=number_of_stacks,
                                          number_of_cells=params.get('number_of_cells'),
                                          size_mean_and_std=params.get('size_mean_and_std'),
                                          equal_dimensions=params.get('equal_dimensions'),
                                          spikiness_range=params.get('spikiness_range'),
                                          spike_size_range=params.get('spike_size_range'),
                                          spike_smoothness_range=params.get('spike_smoothness_range'))
            sim.generate_stacks_batch(params_folder=cell_parameter_folder, outputfolder=inputfolder,
                                      resolution=params.get('input_voxel_size'),
                                      stack_size_microns=params.get('stack_size_microns'),
                                      max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))

        sim.generate_psfs_batch(outputfolder=psffolder, sigmas=params.get('psf_sigmas'),
                                elongations=params.get('psf_elongations'), resolution=params.get('input_voxel_size'),
                                max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
        sim.convolve_batch(inputfolder=inputfolder, psffolder=psffolder, outputfolder=convolution_folder,
                           max_threads=int(params.get('max_threads')/3), print_progress=params.get('print_progress'))
        sim.resize_batch(inputfolder=convolution_folder, outputfolder=resizing_folder,
                         resolutions=params.get('voxel_sizes_for_resizing'),
                         max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
        sim.add_noise_batch(inputfolder=resizing_folder, outputfolder=noise_folder,
                            gaussian_snrs=params.get('gaussian_snrs'), poisson_snrs=params.get('poisson_snrs'),
                            max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))

    if step is None or step == 2:
        dec.deconvolve_batch(inputfolder=noise_folder, outputfolder=deconvolution_folder,
                             algorithm=params.get('algorithm'), rif_lambda=params.get('rif_lambda'),
                             rltv_lambda=params.get('rltv_lambda'), iterations=params.get('iterations'),
                             normalize=params.get('normalize'), perform=params.get('perform'),
                             detect=params.get('detect'), wiener=params.get('wiener'), low=params.get('low'),
                             terminate=params.get('terminate'), log_computing_time=True,
                             logfolder=log_folder, print_progress=params.get('print_progress'))

    if step is None or step == 3:
        quant.segment_batch(inputfolder=deconvolution_folder, outputfolder=segmentation_folder,
                            preprocess=params.get('preprocess'), thr=params.get('thr'),
                            relative_thr=params.get('relative_thr'), postprocess=params.get('postprocess'),
                            max_threads=params.get('max_threads'), print_progress=params.get('print_progress'),
                            log_computing_time=False, logfolder=log_folder)
        quant.compare_to_ground_truth_batch(inputfolder=segmentation_folder, reffolder=inputfolder,
                                            outputfolder=accuracy_folder, max_threads=params.get('max_threads'),
                                            print_progress=params.get('print_progress'))
        quant.extract_metadata(inputfile=accuracy_folder[:-1] + '.csv', default_resolution=params.get('input_voxel_size'))
        quant.combine_log(inputfolder=log_folder)
        quant.extract_metadata(inputfile=log_folder[:-1] + '.csv', default_resolution=params.get('input_voxel_size'))


########################################

default_parameters = pd.Series({'simulation_folder': '../../../Data/test_simulation',
                                'max_threads': 4,
                                'print_progress': True,
                                'cell_parameter_filename': 'cell_parameters.csv',
                                'cell_parameter_folder': 'cell_parameters',
                                'number_of_stacks': None,
                                'number_of_cells': 2,
                                'size_mean_and_std': (10, 2),
                                'equal_dimensions': False,
                                'spikiness_range': (0, 0),
                                'spike_size_range': (0, 0),
                                'spike_smoothness_range': (0.05, 0.1),
                                'inputfolder': 'input',
                                'input_voxel_size': 0.3,
                                'stack_size_microns': [10, 100, 100],
                                'psffolder': 'psf',
                                'psf_sigmas': [0.1, 0.5],
                                'psf_elongations': [3],
                                'convolution_results_folder': 'convolved',
                                'resizing_results_folder': 'resized',
                                'voxel_sizes_for_resizing': [[1, 0.5, 0.5]],
                                'noise_results_folder': 'noise',
                                'poisson_snrs': [None],
                                'gaussian_snrs': [None],
                                'deconvolution_results_folder': 'deconvolved',
                                'algorithm': ['deconvolution_lab_rif', 'deconvolution_lab_rltv'],
                                'rif_lambda': [0.001, 001],
                                'rltv_lambda': 0.001,
                                'iterations': [5, 10],
                                'normalize': True,
                                'perform': True,
                                'detect': True,
                                'wiener': 0.001,
                                'low': 1,
                                'terminate': 0.1,
                                'log_computing_time': True,
                                'segmentation_results_folder': 'segmented',
                                'preprocess': False,
                                'thr': None,
                                'relative_thr': False,
                                'postprocess': False,
                                'log_folder': 'timelog',
                                'accuracy_results_folder': 'accuracy',
                                'dimensions_folder': 'dimensions'
                                })


if __name__ == '__main__':

    args = sys.argv[1:]
    step = None
    if len(args) > 0:
        params = pd.read_csv(args[0], sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze()
        for c in default_parameters.index:
            if c not in params.index:
                params[c] = default_parameters[c]
        if len(args) > 1:
            step = args[1]
            try:
                step = int(step)
            except:
                step = None
            if step not in [1, 2, 3]:
                step = None
    else:
        params = default_parameters
    if step is None:
        print "Running all 3 steps: simulation, deconvolution and quantification"
    elif step == 1:
        print "Running only simulation step"
    elif step == 2:
        print "Running only deconvolution step"
    elif step == 3:
        print "Running only quantification step"
    run_simulation(step, **dict(params))


