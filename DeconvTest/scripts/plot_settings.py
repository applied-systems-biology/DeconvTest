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

    stats = []
    for fn in [accuracy_folder[:-1] + '.csv', log_folder[:-1] + '.csv']:
        stat = pd.read_csv(fn, sep='\t', index_col=0)
        stat = extract_metadata(stat)
        compute_settings_summary(stat, fn[:-4] + '_summary.csv')
        plot_algorithms(stat, plot_folder, cols, labels)
        stats.append(stat)
    best_settings = choose_best_settings(stats[0], col='Jaccard index')
    for stat in stats:
        plot_best_settings(stat, best_settings, plot_folder, cols, labels)


def extract_metadata(stat):
    for i in range(len(stat)):
        name = stat.iloc[i]['Name'].split('/')[-3]
        alg = name.split('_')[0]
        stat.at[i, 'Algorithm'] = alg
        stat.at[i, 'Settings'] = name[len(alg)+1:]
    return stat


def compute_settings_summary(stat, outputfile):

    summary = stat.groupby(['Algorithm', 'Settings']).mean().reset_index()
    summary.to_csv(outputfile, sep='\t')


def plot_algorithms(stat, plot_folder, cols, labels):
    alg = 'DeconvolutionLab-RIF'
    curstat = stat[stat['Algorithm'] == alg]
    if len(curstat) > 0:

        plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='Lambda', rotation=90,
             margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, title=alg, figsize=(5, 4))

    alg = 'DeconvolutionLab-RLTV'
    curstat = stat[stat['Algorithm'] == alg]
    curstat = curstat[curstat['Lambda'] < 10]
    if len(curstat) > 0:
        plot(curstat, cols, plot_folder + alg + '_', x='Lambda', labels=labels, hue='Iterations', rotation=90,
             margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, title=alg, figsize=(5, 4))

    alg = 'IterativeDeconvolve3D'
    curstat = stat[stat['Algorithm'] == alg]
    if len(curstat) > 0:
        hues = ['Normalize PSF', 'Perform anti-ringing', 'Detect divergence']
        xs = ['Low pass filter, pixels', 'Terminate if mean delta <', 'Wiener filter gamma']
        for x in xs:
            for hue in hues:
                plot(curstat, cols, plot_folder + alg + '_', x=x, labels=labels, hue=hue, rotation=90,
                     margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, figsize=(5, 4), title=alg)


def choose_best_settings(stat, col=None):
    if col is None:
        col = 'Jaccard index'
    best_stat = dict()
    for alg in stat['Algorithm'].unique():
        curstat = stat[stat['Algorithm'] == alg]
        curstat = curstat.sort_values(col, ascending=False).reset_index()
        best_stat[alg] = curstat.iloc[0]['Settings']
    return best_stat


def plot_best_settings(stat, setting, plot_folder, cols, labels):
    best_stat = pd.DataFrame()
    for alg in setting.keys():
        best_stat = pd.concat([best_stat, stat[(stat['Algorithm'] == alg)
                                               & (stat['Settings'] == setting[alg])]])

    if 'Computational time' in best_stat.columns:
        plot(best_stat, cols, plot_folder + 'Best_settings_', x='Algorithm', labels=labels, rotation=45,
             margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.35}, logscale=True, figsize=(5, 4))
    else:
        plot(best_stat, cols, plot_folder + 'Best_settings_', x='Algorithm', labels=labels, rotation=45,
             margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.35}, figsize=(5, 4))


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















