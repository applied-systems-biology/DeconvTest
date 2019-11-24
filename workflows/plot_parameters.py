from __future__ import division

import pandas as pd

import sys
import os
import seaborn as sns
import pylab as plt
import numpy as np
from helper_lib import filelib


def plot(stat, columns, outputname, labels=None, logscale=False, dpi=300, xlabel=None, **kwargs):
    """
    Plots given statistics from a given DataFrame.

    Parameters
    ----------
    stat : pandas.DataFrame
        Table containing the statistics to plot.
    columns : list of str
        List of data columns to plot.
    labels : list of str, optional
        Names to lable the y axis instead of column names.
        If None, the column names will be used.
        Default is None.
    outputname : str
        Path to the output directory where to store the plots.
    logscale : bool, optional
        If True, the data will be plotted in a semi-logarithmic scale.
        Default is False.
    dpi : int, optional
        Image resolution.
        Default is 300.
    kwargs : key, value pairings
        Keyword arguments passed to the `seaborn.boxplot` and `seaborn.pointplot` functions.
    """
    filelib.make_folders([os.path.dirname(outputname)])
    rotation = kwargs.pop('rotation', None)
    margins = kwargs.pop('margins', None)
    title = kwargs.pop('title', None)
    x = kwargs.get('x')
    figsize = kwargs.pop('figsize', None)
    hue = kwargs.get('hue', None)
    normalize = kwargs.pop('normalize', True)
    for c in ['SNR', 'SNR2', 'SNR3', 'SNR4']:
        if x == c or hue == c:
            stat[c] = np.array(stat[c]).astype(str)
            for snr in stat[c].unique():
                if snr == 'nan':
                    stat.at[stat[c] == snr, c] = '10000'
            stat[c] = np.array(stat[c]).astype(float).astype(int)
    if hue is not None:
        stat = stat.sort_values([x, hue])
    else:
        stat = stat.sort_values(x)
    if 'SNR' in stat.columns:
        stat.loc[stat[stat['SNR'] == 10000].index, 'SNR'] = 'no noise'
    for i, c in enumerate(columns):
        if c in stat.columns:
            name = c + '_vs_' + x
            if hue is not None:
                name = name + '_and_' + hue
            name = name.replace(' ', '_').replace(',', '')

            st = ['boxplot', 'poitplot']
            func = [sns.boxplot, sns.pointplot]
            for j in range(len(st)):

                if figsize is not None:
                    plt.figure(figsize=figsize)
                ax = func[j](y=c, data=stat, **kwargs)
                if logscale:
                    ax.set_yscale('log')
                if rotation is not None:
                    ax.set_xticklabels(stat[kwargs.get('x')].unique(), rotation=rotation)
                if margins is not None:
                    plt.subplots_adjust(**margins)
                if title is not None:
                    plt.title(title)
                sns.despine()
                if labels is not None:
                    plt.ylabel(labels[i])
                if normalize is True and c == 'Jaccard index':
                    plt.ylim(0, 1)
                if xlabel is not None:
                    plt.xlabel(xlabel)
                plt.savefig(outputname + name + '_' + st[j] + '.png', dpi=dpi)
                plt.savefig(outputname + name + '_' + st[j] + '.svg')
                plt.close()


def summarize_results(**params):

    simulation_folder = params.get('simulation_folder')
    filelib.make_folders([simulation_folder])
    if not simulation_folder.endswith('/'):
        simulation_folder += '/'

    logfolder = simulation_folder + params.get('logfolder')
    if not logfolder.endswith('/'):
        logfolder += '/'

    accuracy_results_folder = simulation_folder + params.get('accuracy_results_folder')
    if not accuracy_results_folder.endswith('/'):
        accuracy_results_folder += '/'

    plotfolder = simulation_folder + params.get('plotfolder')
    if not plotfolder.endswith('/'):
        plotfolder += '/'

    cols = ['RMSE', 'NRMSE', 'Computational time']
    labels = ['RMSE', 'NRMSE', 'Computation time (seconds)']
    titles = {'deconvolution_lab_rif': 'RIF', 'deconvolution_lab_rltv': 'RLTV', 'iterative_deconvolve_3d': 'DAMAS'}

    for fn in [accuracy_results_folder[:-1] + '.csv', logfolder[:-1] + '.csv']:
        stat = pd.read_csv(fn, sep='\t', index_col=0)
        stat = extract_metadata(stat)
        stat['PSF aspect ratio'] = np.array(stat['PSF aspect ratio']).astype(int)
        for alg in stat['Deconvolution algorithm'].unique():
            curstat = stat[stat['Deconvolution algorithm'] == alg].reset_index()

            if 'SNR' in stat.columns:
                if 'SNR2' in stat.columns:
                    plot(curstat, cols, plotfolder + alg + '_', labels=labels,
                         x='SNR', hue='SNR2', title=titles[alg],
                         figsize=(3.5, 3), rotation=90,
                         margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, palette='viridis')

                    curstat = curstat[curstat['SNR2'] > 1000].reset_index()
                curstat['PSF standard deviation in xy, $\mu m$'] = curstat['PSF sigma xy um']
                curstat['Voxel size, $\mu m$'] = curstat['Voxel size']
                plot(curstat, cols, plotfolder + alg + '_', labels=labels,
                     x='SNR', hue='PSF standard deviation in xy, $\mu m$',  title=titles[alg], figsize=(3, 3.5),
                     rotation=90, palette='viridis',
                     margins={'left': 0.23, 'right': 0.95, 'top': 0.9, 'bottom': 0.25})
                plot(curstat, cols, plotfolder + alg + '_', labels=labels,
                     x='SNR', hue='Voxel size, $\mu m$', title=titles[alg], figsize=(3, 3.5), rotation=90,
                     margins={'left': 0.23, 'right': 0.95, 'top': 0.9, 'bottom': 0.25}, palette='viridis')

            else:
                plot(curstat, cols, plotfolder + alg + '_', labels=labels, x='PSF sigma xy um',
                     hue='PSF aspect ratio', title=titles[alg], figsize=(3, 3),
                     xlabel='PSF standard deviation in xy, $\mu m$',
                     rotation=90, margins={'left': 0.23, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, palette='viridis')
                plot(curstat, cols, plotfolder + alg + '_', labels=labels, x='Voxel size', hue='PSF sigma xy um',
                     title=titles[alg], figsize=(3, 3.4), xlabel='Voxel size, $\mu m$',
                     rotation=90, margins={'left': 0.23, 'right': 0.95, 'top': 0.9, 'bottom': 0.35}, palette='viridis')


def extract_metadata(stat):
    for i in range(len(stat)):
        name = stat.iloc[i]['Name'].split('/')[-3]
        alg = name.split('_')[0]
        stat.at[i, 'Algorithm'] = alg
        stat.at[i, 'Settings'] = name[len(alg)+1:]
    return stat


########################################


default_parameters = dict({'simulation_folder': 'test_simulation',
                           'accuracy_results_folder': 'accuracy_measures',
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















