from __future__ import division

import sys
import pandas as pd

from helper_lib import filelib
from DeconvTest.classes.plotting import plot


def summarize_results(**params):

    simulation_folder = params.get('simulation_folder')
    filelib.make_folders([simulation_folder])
    if not simulation_folder.endswith('/'):
        simulation_folder += '/'

    log_folder = simulation_folder + params.get('log_folder')
    if not log_folder.endswith('/'):
        log_folder += '/'

    accuracy_folder = simulation_folder + params.get('accuracy_results_folder')
    if not accuracy_folder.endswith('/'):
        accuracy_folder += '/'

    dimensions_folder = simulation_folder + params.get('dimensions_folder')
    if not dimensions_folder.endswith('/'):
        dimensions_folder += '/'

    plot_folder = simulation_folder + params.get('plot_folder')
    if not plot_folder.endswith('/'):
        plot_folder += '/'

    cols = ['Overdetection error', 'Underdetection error', 'Overlap error',
            'Jaccard index', 'Sensitivity', 'Precision', 'Computational time']
    labels = ['Overdetection error', 'Underdetection error', 'Overlap error',
              'Jaccard index', 'Sensitivity', 'Precision', 'Computational time']

    for fn in [accuracy_folder[:-1] + '.csv', log_folder[:-1] + '.csv']:
        stat = pd.read_csv(fn, sep='\t', index_col=0)
        stat = extract_metadata(stat)
        for alg in stat['Algorithm'].unique():
            curstat = stat[stat['Algorithm'] == alg]
            plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='Poisson_SNR', hue='Gaussian_SNR',
                 title=alg, figsize=(5, 4),
                 rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})

            curstat = curstat[curstat['Gaussian_SNR'] > 1000]
            plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='Poisson_SNR', hue='PSF sigma X',
                 title=alg, figsize=(5, 4),
                 rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})
            plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='Poisson_SNR', hue='PSF elongation',
                 title=alg, figsize=(5, 4),
                 rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})
            plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='Poisson_SNR', hue='Resolution',
                 title=alg, figsize=(5, 4),
                 rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})
            plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='PSF sigma X', hue='PSF elongation',
                 title=alg, figsize=(5, 4),
                 rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})
            plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='Resolution', hue='PSF sigma X',
                 title=alg, figsize=(5, 3.5),
                 rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.35})


def extract_metadata(stat):
    for i in range(len(stat)):
        name = stat.iloc[i]['Name'].split('/')[-3]
        alg = name.split('_')[0]
        stat.at[i, 'Algorithm'] = alg
        stat.at[i, 'Settings'] = name[len(alg)+1:]
    return stat


########################################


default_parameters = pd.Series({'simulation_folder': '../../../Data/test_simulation',
                                'log_folder': 'timelog',
                                'accuracy_results_folder': 'accuracy',
                                'dimensions_folder': 'dimensions',
                                'plot_folder': 'plots'
                                })


if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) > 0:
        params = pd.read_csv(args[0], sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze()
        for c in default_parameters.index:
            if c not in params.index:
                params[c] = default_parameters[c]
        summarize_results(**dict(params))















