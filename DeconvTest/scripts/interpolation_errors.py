from __future__ import division

import sys
import pandas as pd
from helper_lib import filelib

from DeconvTest.batch import simulation as sim
from DeconvTest.batch import quantification as quant
from DeconvTest.classes.plotting import plot


def run_simulation(**params):
    simulation_folder = params.get('simulation_folder')
    filelib.make_folders([simulation_folder])
    if not simulation_folder.endswith('/'):
        simulation_folder += '/'

    cell_parameter_filename = simulation_folder + params.get('cell_parameter_filename')
    if not cell_parameter_filename.endswith('.csv'):
        cell_parameter_filename += '.csv'

    inputfolder = simulation_folder + params.get('inputfolder')
    if not inputfolder.endswith('/'):
        inputfolder += '/'

    accuracy_folder = simulation_folder + params.get('accuracy_results_folder')
    if not accuracy_folder.endswith('/'):
        accuracy_folder += '/'

    plotting_folder = simulation_folder + params.get('plotting_results_folder')
    if not plotting_folder.endswith('/'):
        plotting_folder += '/'

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

    orders = [0, 1, 2, 3, 4]
    thresholds = [0, 0.99, None]

    for order in orders:
        folder_order = 'order=' + str(order) + '/'
        sim.resize_batch(inputfolder=inputfolder, outputfolder=simulation_folder + 'downsized/' + folder_order,
                         resolutions=params.get('voxel_sizes_for_resizing'), order=order,
                         max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))
        sim.resize_batch(inputfolder=simulation_folder + 'downsized/' + folder_order,
                         outputfolder=simulation_folder + 'upsized/' + folder_order,
                         resolutions=[params.get('input_voxel_size')], order=order,
                         max_threads=params.get('max_threads'), print_progress=params.get('print_progress'))

        for thr in thresholds:
            folder_thr = 'order=' + str(order) + '_threshold=' + str(thr) + '/'
            quant.segment_batch(inputfolder=simulation_folder + 'upsized/' + folder_order,
                                outputfolder=simulation_folder + 'upsized_segmented/' + folder_thr,
                                preprocess=False, thr=thr,
                                relative_thr=True, postprocess=False,
                                max_threads=params.get('max_threads'), print_progress=params.get('print_progress'),
                                log_computing_time=False)

    quant.compare_to_ground_truth_batch(inputfolder=simulation_folder + 'upsized_segmented/',
                                        reffolder=inputfolder,
                                        outputfolder=accuracy_folder,
                                        max_threads=params.get('max_threads')/3,
                                        print_progress=params.get('print_progress'))

    stat = pd.read_csv(accuracy_folder[:-1] + '.csv', sep='\t', index_col=0)
    for i in range(len(stat)):
        fn = stat.iloc[i]['Name']
        thr = fn.split('threshold=')[1].split('/')[0]
        order = fn.split('order=')[1].split('_')[0]
        if thr == 'None':
            thr = 'Otsu'
        stat.at[i, 'Interpolation order'] = order
        stat.at[i, 'Threshold'] = thr
    stat.to_csv(accuracy_folder[:-1] + '.csv', sep='\t')

    plot(stat=stat, columns=['Overdetection error', 'Overlap error', 'Underdetection error'],
         outputname=plotting_folder, logscale=True, x='Interpolation order', hue='Threshold')
    plot(stat=stat, columns=['Jaccard index', 'Precision', 'Sensitivity'],
         outputname=plotting_folder, logscale=False, x='Interpolation order', hue='Threshold')


####################################

params = pd.Series({'simulation_folder': './Data/interpolation_errors',
                    'max_threads': 5,
                    'print_progress': True,
                    'cell_parameter_filename': 'cell_parameters.csv',
                    'number_of_stacks': None,
                    'number_of_cells': 10,
                    'size_mean_and_std': (7.5, 1),
                    'equal_dimensions': True,
                    'spikiness_range': (0, 0),
                    'spike_size_range': (0, 0),
                    'spike_smoothness_range': (0.05, 0.1),
                    'inputfolder': 'input',
                    'input_voxel_size': 0.01,
                    'resizing_results_folder': 'downsized',
                    'voxel_sizes_for_resizing': [0.3],
                    'accuracy_results_folder': 'accuracy',
                    'plotting_results_folder': 'plots',
                    'variables_to_plot': [('PSF sigma X', 'PSF elongation'), ('resolution', 'PSF sigma X')]
                    })

if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) == 2:
        params['simulation_folder'] = args[0]
        if args[1] == '-test':
            params['input_voxel_size'] = 0.2
            run_simulation(**dict(params))
        elif args[1] == '-run':
            run_simulation(**dict(params))

