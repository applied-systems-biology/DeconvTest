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

                mean_err = []
                for cur_x in stat[x].unique():
                    curstat = stat[stat[x] == cur_x]
                    mean_err.append(np.mean(curstat[c]))
                min_ind = np.argmin(mean_err)
                minval = np.min(stat[c])
                maxval = np.max(stat[c])
                plt.plot([min_ind - 0.5, min_ind + 0.5, min_ind + 0.5, min_ind - 0.5, min_ind - 0.5],
                         [maxval*1.05, maxval*1.05, minval*0.8, minval*0.8, maxval*1.05], 'k',
                         linewidth=0.5, linestyle='--')


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
    labels = ['RMSE', 'NRMSE', 'Computational time, seconds']

    pairs = [['SNR', 'Deconvolution algorithm'], ['Voxel size', 'Deconvolution algorithm'],
             ['PSF sigma xy um', 'Deconvolution algorithm'], ['PSF aspect ratio', 'Deconvolution algorithm']]

    fn = accuracy_results_folder[:-1] + '.csv'
    stat = pd.read_csv(fn, sep='\t', index_col=0)
    stat = extract_metadata(stat)
    #
    # for pair in pairs:
    #     stat = stat.sort_values(pair)
    #     plot(stat, cols, plotfolder + 'average/', labels=labels, x=pair[0], hue=pair[1], figsize=(5, 4),
    #          rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})

    best_stat = choose_best_settings(stat, col='NRMSE')
    best_stat.to_csv(fn[:-4] + '_best_settings.csv', sep='\t')
    # plot_best_settings(best_stat, plotfolder + 'best_settings/')
    plot_details(stat, plotfolder, cols, cols)

    # fn = logfolder[:-1] + '.csv'
    # stat = pd.read_csv(fn, sep='\t', index_col=0)
    # stat = extract_metadata(stat)
    # for pair in pairs:
    #     stat = stat.sort_values(pair)
    #     plot(stat, cols, plotfolder + 'average/', labels=cols, x=pair[0], hue=pair[1], logscale=True,
    #          rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, figsize=(5, 4))


def choose_best_settings(stat, col=None):
    if col is None:
        col = 'NRMSE'
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

                        paramsstat = paramsstat.sort_values(col, ascending=True).reset_index()
                        paramsstat.at[:, 'PSF size'] = 'XY=' + str(sigma) + '$\mu m$; aspect=' + str(elon)
                        paramsstat.at[:, col + ' variation minmax'] = paramsstat.iloc[0][col] - paramsstat.iloc[-1][col]
                        paramsstat.at[:, col + ' variation 1-2'] = paramsstat.iloc[0][col] - paramsstat.iloc[1][col]
                        best_stat = pd.concat([best_stat, paramsstat.iloc[0:1]])

    best_stat = best_stat.reset_index(drop=True)
    return best_stat


def plot_best_settings(stat, plot_folder):
    cols = ['NRMSE variation minmax', 'NRMSE variation 1-2']
    titles = {'deconvolution_lab_rif': 'RIF', 'deconvolution_lab_rltv': 'RLTV', 'iterative_deconvolve_3d': 'DAMAS'}

    for resolution in stat['Voxel size'].unique():
        paramsstat = stat[stat['Voxel size'] == resolution]
        res = str(resolution).replace('[', '').replace(']', '').replace(' ', '_').replace('.', 'p').replace('__', '_')

        ##################################
        alg = 'deconvolution_lab_rif'
        curstat = paramsstat[paramsstat['Deconvolution algorithm'] == alg]
        if len(curstat) > 0:
            plot(curstat, ['regularization_lambda'], plot_folder + alg + '_voxel_size=' + res + '_',
                 x='PSF size', hue='SNR', rotation=90, figsize=(4, 6), logscale=True, labels=['$\lambda$'],
                 title=titles[alg], normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35},
                 palette='viridis')
            plot(curstat, cols, plot_folder + alg + '_voxel_size=' + res + '_',
                 x='PSF size', hue='SNR', rotation=90, figsize=(4, 6),
                 title=titles[alg], normalize=True, margins={'left': 0.25, 'right': 0.95, 'top': 0.9, 'bottom': 0.35},
                 palette='viridis')

        ##################################
        alg = 'deconvolution_lab_rltv'
        curstat = paramsstat[paramsstat['Deconvolution algorithm'] == alg]
        if len(curstat) > 0:
            plot(curstat, ['regularization_lambda'], plot_folder + alg + '_voxel_size=' + res + '_',
                 x='PSF size', hue='SNR', rotation=90, figsize=(4, 6), logscale=True, labels=['$\lambda$'],
                 title=titles[alg], normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35},
                 palette='viridis')
            plot(curstat, ['iterations'] + cols, plot_folder + alg + '_voxel_size=' + res + '_',
                 x='PSF size', hue='SNR', rotation=90, figsize=(4, 6),
                 title=titles[alg], normalize=True, margins={'left': 0.25, 'right': 0.95, 'top': 0.9, 'bottom': 0.35},
                 palette='viridis')


        ##################################
        alg = 'iterative_deconvolve_3d'
        curstat = paramsstat[paramsstat['Deconvolution algorithm'] == alg]
        if len(curstat) > 0:
            plot(curstat, ['normalize', 'perform', 'detect', 'low', 'terminate', 'wiener'] + cols,
                 plot_folder + alg + '_voxel_size=' + res + '_',
                 x='PSF size', hue='SNR', rotation=90, figsize=(4, 6),
                 labels=['Normalize PSF', 'Perform anti-ringing', 'Detect divergence', 'Low pass filter, pixels',
                         'Terminate if mean delta <', 'Wiener filter gamma'] + cols,
                 title=titles[alg], normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35},
                 palette='viridis')


def plot_details(stat, plot_folder, labels, cols):
    titles = {'deconvolution_lab_rif': 'RIF', 'deconvolution_lab_rltv': 'RLTV', 'iterative_deconvolve_3d': 'DAMAS'}
    stat['SNR'] = np.array(stat['SNR']).astype(str)
    for snr in stat['SNR'].unique():
        if snr == 'nan':
            stat.at[stat['SNR'] == snr, 'SNR'] = 'no noise'
    for sigma in stat['PSF sigma xy um'].unique():
        for elon in stat['PSF aspect ratio'].unique():
            for res in stat['Voxel size'].unique():
                for snr in stat['SNR'].unique():
                    paramsstat = stat[(stat['PSF sigma xy um'] == sigma) & (stat['PSF aspect ratio'] == elon)
                                      & (stat['Voxel size'] == res) & (stat['SNR'] == snr)]
                    current_plot_folder = plot_folder + 'detailed/sigma_xy=' + str(sigma).replace('.', 'p') \
                                          + '_aspect_ratio=' + str(elon).replace('.', 'p') \
                                          + '_voxel_size=' + str(res).replace('[', '').replace(']', '')\
                                              .replace(' ', '_').replace('.', 'p').replace('__', '_') \
                                          + '_snr=' + str(snr).replace(' ', '_') + '/'


                    ##################################
                    alg = 'deconvolution_lab_rif'
                    curstat = paramsstat[paramsstat['Deconvolution algorithm'] == alg]
                    if len(curstat) > 0:
                        plot(curstat, cols, current_plot_folder + alg + '_', labels=labels, x='regularization_lambda',
                             rotation=90, figsize=(2.2, 3), title=titles[alg], normalize=True, xlabel='$\lambda$',
                             margins={'left': 0.3, 'right': 0.95, 'top': 0.9, 'bottom': 0.28}, color='gray')

                    ##################################
                    alg = 'deconvolution_lab_rltv'
                    curstat = paramsstat[paramsstat['Deconvolution algorithm'] == alg]
                    if len(curstat) > 0:
                        plot(curstat, cols, current_plot_folder + alg + '_', x='regularization_lambda', labels=labels,
                             rotation=90, title=titles[alg], figsize=(2.2, 3), normalize=True,
                             margins={'left': 0.3, 'right': 0.95, 'top': 0.9, 'bottom': 0.28}, color='gray',
                             xlabel='$\lambda$')


                    ##################################
                    alg = 'iterative_deconvolve_3d'
                    curstat = paramsstat[paramsstat['Deconvolution algorithm'] == alg]
                    curstat['low'] = np.array(curstat['low']).astype(int)
                    if len(curstat) > 0:
                        xs = ['low', 'terminate', 'wiener']
                        nxs = ['Low pass filter, pixels', 'Terminate if mean delta <', 'Wiener filter gamma']
                        for j, x in enumerate(xs):
                            plot(curstat, cols, current_plot_folder + alg + '_', x=x, xlabel=nxs[j],
                                 labels=labels, rotation=90, title=titles[alg], figsize=(2.2, 3), normalize=True,
                                 margins={'left': 0.32, 'right': 0.95, 'top': 0.9, 'bottom': 0.28}, color='gray')


def extract_metadata(stat):
    for i in range(len(stat)):
        name = stat.iloc[i]['Name'].split('/')[-3]
        alg = name.split('_')[0]
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















