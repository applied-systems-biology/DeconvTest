from __future__ import division

import sys
import os
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
    sp_stat = plot_accuracy_against_number_of_cells(stat, simulation_folder + params.get('cell_parameter_folder'),
                                                    plot_folder + 'accuracy_vs_number_of_cells/', ['Jaccard index'])
    sp_stat.to_csv(fn[:-4] + '_cell_metadata.csv', sep='\t')


def plot_accuracy_against_number_of_cells(stat, cell_params_folder, plot_folder, cols):
    files = os.listdir(cell_params_folder)
    cell_params = pd.DataFrame()
    stat = stat.groupby(['Name']).mean()
    stat['Name'] = stat.index
    stat = stat.reset_index(drop=True)
    for fn in files:
        cell_stat = pd.read_csv(cell_params_folder + '/' + fn, sep='\t')
        cell_params = pd.concat([cell_params, pd.DataFrame({'StackName': [fn],
                                                            'Number_of_cells': [len(cell_stat)]})])

    for i in range(len(stat)):
        stat.loc[i, 'StackName'] = stat.iloc[i]['Name'].split('/')[-1][:-4] + '.csv'
        stat.loc[i, 'Group'] = stat.iloc[i]['Name'].split('/')[-3] + '_' + stat.iloc[i]['Name'].split('/')[-2] + '_'

    stat = stat.merge(cell_params, how='left', on='StackName')
    for gr in stat['Group'].unique():
        curstat = stat[stat['Group'] == gr]
        plot_lmplot(curstat, cols, plot_folder + gr, labels=cols, x='Number_of_cells', figsize=(5, 4),
                    margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.1})

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















