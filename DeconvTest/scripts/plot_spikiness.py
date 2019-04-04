from __future__ import division

import sys
import re
import pandas as pd

from helper_lib import filelib
from DeconvTest.classes.plotting import plot_lmplot


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

    fn = accuracy_folder[:-1] + '.csv'
    stat = pd.read_csv(fn, sep='\t', index_col=0)
    sp_stat = plot_accuracy_against_spikiness(stat, simulation_folder+params.get('cell_parameter_filename'),
                                              plot_folder + 'accuracy_vs_spikiness/', ['Jaccard index'])
    sp_stat.to_csv(fn[:-4] + '_cell_metadata.csv', sep='\t')


def plot_accuracy_against_spikiness(stat, cell_params_file, plot_folder, cols):
    cell_params = pd.read_csv(cell_params_file, sep='\t')
    cell_params['CellID'] = cell_params.index
    p = re.compile('\d*\.*\d+')

    for i in range(len(stat)):
        stat.loc[i, 'CellID'] = int(float(p.findall(stat.iloc[i]['Name'])[-1]))
        stat.loc[i, 'Group'] = stat.iloc[i]['Name'].split('/')[-3] + '/' + stat.iloc[i]['Name'].split('/')[-2] + '/'

    stat = stat.merge(cell_params, how='left', on='CellID')
    for gr in stat['Group'].unique():
        curstat = stat[stat['Group'] == gr]
        plot_lmplot(curstat, cols, plot_folder + gr, labels=cols, x='spikiness', figsize=(5, 4),
                    margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.1})
        plot_lmplot(curstat, cols, plot_folder + gr, labels=cols, x='spike_smoothness', figsize=(5, 4),
                    margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.1})
        plot_lmplot(curstat, cols, plot_folder + gr, labels=cols, x='spike_size',
                    figsize=(5, 4), margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.1})

    return stat


########################################


default_parameters = pd.Series({'simulation_folder': '../../../Data/test_simulation',
                                'log_folder': 'timelog',
                                'accuracy_results_folder': 'accuracy',
                                'dimensions_folder': 'dimensions',
                                'plot_folder': 'plots/'
                                })


if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) > 0:
        params = pd.read_csv(args[0], sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze()
        for c in default_parameters.index:
            if c not in params.index:
                params[c] = default_parameters[c]
        summarize_results(**dict(params))















