from __future__ import division

import sys
import pandas as pd

from helper_lib import filelib
from DeconvTest.modules.plotting import plot


def summarize_results(**params):

    simulation_folder = params.get('simulation_folder')
    filelib.make_folders([simulation_folder])
    if not simulation_folder.endswith('/'):
        simulation_folder += '/'

    logfolder = simulation_folder + params.get('logfolder')
    if not logfolder.endswith('/'):
        logfolder += '/'

    binary_accuracy_results_folder = simulation_folder + params.get('binary_accuracy_results_folder')
    if not binary_accuracy_results_folder.endswith('/'):
        binary_accuracy_results_folder += '/'

    plotfolder = simulation_folder + params.get('plotfolder')
    if not plotfolder.endswith('/'):
        plotfolder += '/'

    cols = ['Overdetection error', 'Underdetection error', 'Overlap error',
            'Jaccard index', 'Sensitivity', 'Precision', 'Computational time']

    pairs = [['SNR', 'Deconvolution algorithm'], ['Voxel size', 'Deconvolution algorithm'],
             ['PSF sigma xy um', 'Deconvolution algorithm'], ['PSF aspect ratio', 'Deconvolution algorithm']]

    fn = binary_accuracy_results_folder[:-1] + '.csv'
    stat = pd.read_csv(fn, sep='\t', index_col=0)
    stat = extract_metadata(stat)

    for pair in pairs:
        stat = stat.sort_values(pair)
        plot(stat, cols, plotfolder + 'average/', labels=cols, x=pair[0], hue=pair[1], figsize=(5, 4),
             rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})

    best_stat = choose_best_settings(stat, col='Jaccard index')
    best_stat.to_csv(fn[:-4] + '_best_settings.csv', sep='\t')

    fn = logfolder[:-1] + '.csv'
    stat = pd.read_csv(fn, sep='\t', index_col=0)
    stat = extract_metadata(stat)
    for pair in pairs:
        stat = stat.sort_values(pair)
        plot(stat, cols, plotfolder + 'average/', labels=cols, x=pair[0], hue=pair[1], logscale=True,
             rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, figsize=(5, 4))


def choose_best_settings(stat, col=None):
    if col is None:
        col = 'Jaccard index'
    stat_summary = stat.groupby(['PSF sigma xy um', 'PSF aspect ratio', 'Voxel size',
                                 'SNR', 'Deconvolution algorithm', 'Settings']).mean().reset_index()
    best_stat = pd.DataFrame()
    for sigma in stat_summary['PSF sigma xy um'].unique():
        for elon in stat_summary['PSF aspect ratio'].unique():
            for res in stat_summary['Voxel size'].unique():
                for snr in stat_summary['SNR'].unique():
                    for alg in stat_summary['Deconvolution algorithm'].unique():
                        paramsstat = stat_summary[(stat_summary['PSF sigma xy um'] == sigma)
                                                  & (stat_summary['PSF aspect ratio'] == elon)
                                                  & (stat_summary['Voxel size'] == res)
                                                  & (stat_summary['SNR'] == snr)
                                                  & (stat_summary['Deconvolution algorithm'] == alg)]

                        paramsstat = paramsstat.sort_values(col, ascending=False).reset_index()
                        paramsstat.at[:, 'PSF size'] = 'XY=' + str(sigma) + '$\mu m$; aspect=' + str(elon)
                        paramsstat.at[:, 'Jaccard index variation minmax'] = paramsstat.iloc[0]['Jaccard index'] \
                                                                             - paramsstat.iloc[-1]['Jaccard index']
                        paramsstat.at[:, 'Jaccard index variation 1-2'] = paramsstat.iloc[0]['Jaccard index'] \
                                                                             - paramsstat.iloc[1]['Jaccard index']
                        best_stat = pd.concat([best_stat, paramsstat.iloc[0:1]])

    best_stat = best_stat.reset_index(drop=True)
    return best_stat


def extract_metadata(stat):
    for i in range(len(stat)):
        name = stat.iloc[i]['Name'].split('/')[-3]
        alg = name.split('_')[0]
        stat.at[i, 'Settings'] = name[len(alg)+1:]
    return stat


########################################

default_parameters = dict({'simulation_folder': 'test_simulation',
                           'binary_accuracy_results_folder': 'binary_accuracy_measures',
                           'logfolder': 'timelog',
                           'plotfolder': 'plots'
                            })


if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) > 0:
        params = pd.read_csv(args[0], sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze()
        for c in default_parameters:
            if c not in params.index:
                params[c] = default_parameters[c]
        summarize_results(**dict(params))















