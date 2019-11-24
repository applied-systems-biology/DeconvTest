from __future__ import division

import sys
import os
import pandas as pd
import seaborn as sns
import pylab as plt
import numpy as np
from scipy.stats import wilcoxon

from helper_lib import filelib


def pvalue_to_star(pvalue, sym='*'):

    if pvalue < 0.001:
        return sym*3
    elif pvalue < 0.01:
        return sym*2
    elif pvalue < 0.05:
        return sym
    else:
        return 'n.s.'


def plot(stat, columns, outputname, labels=None, logscale=False, dpi=300, xlabel=None, opt_stat=False,
         group_stat=False, **kwargs):
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
        stat = stat.sort_values([x, hue, 'SNR', 'Voxel size', 'PSF sigma xy um', 'PSF aspect ratio', 'CellID'])
    else:
        stat = stat.sort_values([x, 'SNR', 'Voxel size', 'PSF sigma xy um', 'PSF aspect ratio', 'CellID'])
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

                if opt_stat:
                    test_stat = []
                    mean_err = []
                    for cur_x in stat[x].unique():
                        curstat = stat[stat[x] == cur_x]
                        test_stat.append(curstat)
                        mean_err.append(np.mean(curstat[c]))
                    min_ind = np.argmin(mean_err)

                    for icurstat, curstat in enumerate(test_stat):
                        if icurstat != min_ind:
                            if len(curstat) == len(test_stat[min_ind]):
                                if st[j] == 'boxplot':
                                    max_c = np.max(curstat[c])
                                else:
                                    max_c = np.mean(curstat[c]) + np.std(curstat[c])
                                pval = wilcoxon(curstat[c], test_stat[min_ind][c])[1]
                                plt.text(icurstat, max_c*1.05, pvalue_to_star(pval), family='sans-serif', fontsize=6,
                                         horizontalalignment='center', verticalalignment='bottom', color='black')

                if group_stat:
                    groups = stat[x].unique()
                    maxval = np.max(stat[c])
                    shift = 0
                    step = 0.05
                    if logscale:
                        step = 0.5
                        shift = 0.4
                    for x_ind1 in range(len(groups)):
                        for x_ind2 in range(x_ind1 + 1, len(groups)):
                            curstat1 = stat[stat[x] == groups[x_ind1]]
                            curstat2 = stat[stat[x] == groups[x_ind2]]
                            pval = wilcoxon(curstat1[c], curstat2[c])[1]
                            y = maxval*1.1 + shift*maxval*step
                            plt.plot([x_ind1, x_ind2], [y, y], 'k', linewidth=0.5)
                            plt.text((x_ind1 + x_ind2)/2, y, pvalue_to_star(pval),
                                     family='sans-serif', fontsize=6,
                                     horizontalalignment='center', verticalalignment='bottom', color='black')
                            shift += 1

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

    stats = []
    for fn in [accuracy_results_folder[:-1] + '.csv', logfolder[:-1] + '.csv']:
        stat = pd.read_csv(fn, sep='\t', index_col=0)
        stat = extract_metadata(stat)
        compute_settings_summary(stat, fn[:-4] + '_summary.csv')
        plot_algorithms(stat, plotfolder, cols, labels)
        stats.append(stat)
    if not os.path.exists(accuracy_results_folder[:-1] + '_best_settings.csv'):
        best_settings = choose_best_settings(stats[0], col='NRMSE')
        pd.Series(best_settings).to_csv(accuracy_results_folder[:-1] + '_best_settings.csv', sep='\t', header=False)
    best_settings = dict(pd.read_csv(accuracy_results_folder[:-1] + '_best_settings.csv',
                                     sep='\t', index_col=0, header=-1).transpose().iloc[0])

    for stat in stats:
        plot_best_settings(stat, best_settings, plotfolder, cols, labels)


def extract_metadata(stat):
    for i in range(len(stat)):
        name = stat.iloc[i]['Name'].split('/')[-3]
        alg = name.split('_')[0]
        stat.at[i, 'Settings'] = name[len(alg)+1:]
    return stat


def compute_settings_summary(stat, outputfile):

    summary = stat.groupby(['Deconvolution algorithm', 'Settings']).mean().reset_index()
    summary.to_csv(outputfile, sep='\t')
    return summary


def plot_algorithms(stat, plot_folder, cols, labels):

    ##################################
    alg = 'deconvolution_lab_rif'

    stat.at[stat[stat['normalize'] == True].index, 'normalize'] = 'PSF normalization'
    stat.at[stat[stat['normalize'] == False].index, 'normalize'] = 'No PSF normalization'

    stat.at[stat[stat['detect'] == True].index, 'detect'] = 'Divergence detection'
    stat.at[stat[stat['detect'] == False].index, 'detect'] = 'No divergence detection'

    stat.at[stat[stat['perform'] == True].index, 'perform'] = 'Anti-ringing'
    stat.at[stat[stat['perform'] == False].index, 'perform'] = 'No anti-ringinig'

    curstat = stat[stat['Deconvolution algorithm'] == alg]
    if len(curstat) > 0:

        plot(curstat, cols, plot_folder + alg + '_', labels=labels, x='regularization_lambda', rotation=90,
             margins={'left': 0.22, 'right': 0.95, 'top': 0.9, 'bottom': 0.3}, title='RIF', figsize=(3, 3),
             xlabel='$\lambda$', color='gray', opt_stat=False)

    ##################################
    alg = 'deconvolution_lab_rltv'
    curstat = stat[stat['Deconvolution algorithm'] == alg]
    curstat['iterations'] = np.array(curstat['iterations']).astype(int)
    if 'NRMSE' in curstat.columns:
        curstat = curstat[curstat['NRMSE'] >= 0]
    if len(curstat) > 0:
        plot(curstat, cols, plot_folder + alg + '_', x='regularization_lambda', labels=labels,
             hue='iterations', rotation=90, margins={'left': 0.22, 'right': 0.95, 'top': 0.9, 'bottom': 0.3},
             title='RLTV', figsize=(3, 3), xlabel='$\lambda$', palette='viridis', opt_stat=False)

    ##################################
    alg = 'iterative_deconvolve_3d'
    curstat = stat[stat['Deconvolution algorithm'] == alg]
    curstat['low'] = np.array(curstat['low']).astype(int)
    if len(curstat) > 0:
        hues = ['normalize', 'perform', 'detect']
        nhues = ['Normalize PSF', 'Perform anti-ringing', 'Detect divergence']
        xs = ['low', 'terminate', 'wiener']
        nxs = ['Low pass filter, pixels', 'Termination parameter', 'Wiener filter gamma']
        for i, hue in enumerate(hues):
            curstat[nhues[i]] = curstat[hue]
            for j, x in enumerate(xs):
                plot(curstat, cols, plot_folder + alg + '_', x=x, labels=labels, hue=nhues[i], rotation=90,
                     margins={'left': 0.25, 'right': 0.95, 'top': 0.9, 'bottom': 0.3}, figsize=(3, 3),
                     title='DAMAS', color='gray', xlabel=nxs[j], opt_stat=False)


def choose_best_settings(stat, col=None):
    summary = stat.groupby(['Deconvolution algorithm', 'Settings']).mean().reset_index()
    if col is None:
        col = 'NRMSE'
    best_stat = dict()
    for alg in summary['Deconvolution algorithm'].unique():
        curstat = summary[summary['Deconvolution algorithm'] == alg]
        curstat = curstat.sort_values(col, ascending=True).reset_index()
        best_stat[alg] = curstat.iloc[0]['Settings']
    return best_stat


def plot_best_settings(stat, setting, plot_folder, cols, labels):
    best_stat = pd.DataFrame()
    for alg in setting.keys():
        best_stat = pd.concat([best_stat, stat[(stat['Deconvolution algorithm'] == alg)
                                               & (stat['Settings'] == setting[alg])]])

    best_stat.loc[best_stat[best_stat['Deconvolution algorithm'] == 'deconvolution_lab_rif'].index,
                  'Deconvolution algorithm'] = 'RIF'
    best_stat.loc[best_stat[best_stat['Deconvolution algorithm'] == 'deconvolution_lab_rltv'].index,
                  'Deconvolution algorithm'] = 'RLTV'
    best_stat.loc[best_stat[best_stat['Deconvolution algorithm'] == 'iterative_deconvolve_3d'].index,
                  'Deconvolution algorithm'] = 'DAMAS'
    plot(best_stat, cols, plot_folder + 'Best_settings_', x='Deconvolution algorithm', labels=labels, rotation=0,
         margins={'left': 0.31, 'right': 0.95, 'top': 0.93, 'bottom': 0.1}, figsize=(2, 3), color='gray', xlabel='',
         group_stat=True)
    plot(best_stat, ['Computational time'], plot_folder + 'Best_settings_', x='Deconvolution algorithm',
         labels=['Computation time'], rotation=0,
         margins={'left': 0.31, 'right': 0.95, 'top': 0.93, 'bottom': 0.1}, figsize=(2, 3), color='gray', xlabel='',
         group_stat=True, logscale=True)


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















