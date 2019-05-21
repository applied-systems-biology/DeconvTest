from __future__ import division

import os
import seaborn as sns
import pylab as plt
import numpy as np
from scipy.stats import linregress

from helper_lib import filelib


def plot(stat, columns, outputname, labels=None, logscale=False, dpi=300, **kwargs):
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
    for i, c in enumerate(columns):
        if c in stat.columns:
            name = c + '_vs_' + x
            if hue is not None:
                name = name + '_and_' + hue
            name = name.replace(' ', '_').replace(',', '')

            if figsize is not None:
                plt.figure(figsize=figsize)
            ax = sns.boxplot(y=c, data=stat, **kwargs)
            if logscale:
                ax.set_yscale('log')
            if rotation is not None:
                ax.set_xticklabels(stat[x].unique(), rotation=rotation)
            if margins is not None:
                plt.subplots_adjust(**margins)
            if title is not None:
                plt.title(title)
            sns.despine()
            if labels is not None:
                plt.ylabel(labels[i])
            if normalize is True and c == 'Jaccard index':
                plt.ylim(0, 1)
            plt.savefig(outputname + name + '_boxplot.png', dpi=dpi)
            plt.savefig(outputname + name + '_boxplot.svg')
            plt.close()

            if figsize is not None:
                plt.figure(figsize=figsize)
            ax = sns.pointplot(y=c, data=stat, **kwargs)
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
            plt.savefig(outputname + name + '_pointplot.png', dpi=dpi)
            plt.savefig(outputname + name + '_pointplot.svg')
            plt.close()


def plot_lmplot(stat, columns, outputname, labels=None, logscale=False, dpi=300, **kwargs):
    """
    Plots a linear model plot from given statistics from a given DataFrame.

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
    margins = kwargs.pop('margins', None)
    title = kwargs.pop('title', None)
    x = kwargs.get('x')
    figsize = kwargs.pop('figsize', None)
    hue = kwargs.get('hue', None)
    normalize = kwargs.pop('normalize', True)
    if hue is not None:
        stat = stat.sort_values([x, hue])
    else:
        stat = stat.sort_values(x)
    for i, c in enumerate(columns):
        if c in stat.columns:
            name = c + '_vs_' + x
            if hue is not None:
                name = name + '_and_' + hue
            name = name.replace(' ', '_').replace(',', '')

            if figsize is not None:
                plt.figure(figsize=figsize)
            ax = sns.lmplot(y=c, data=stat, **kwargs)
            if logscale:
                ax.set_yscale('log')
            if margins is not None:
                plt.subplots_adjust(**margins)
            if title is not None:
                plt.title(title)
            sns.despine()
            if labels is not None:
                plt.ylabel(labels[i])
            if normalize is True and c == 'Jaccard index':
                plt.ylim(0, 1)

            rvalue, pvalue = linregress(stat[kwargs.get('x')], stat[c])[2:4]
            rvalue = round(rvalue, 2)
            if round(pvalue, 2) > 0:
                pvalue = '{:.2f}'.format(pvalue)
            else:
                pvalue = '{:.1e}'.format(pvalue)
            plt.text(0.8*stat[kwargs.get('x')].max(), 0.1, 'r = {:.2f}\np = {}'.format(rvalue, pvalue),
                     fontsize=10, horizontalalignment='left', verticalalignment='bottom')
            plt.savefig(outputname + name + '.png', dpi=dpi)
            plt.savefig(outputname + name + '.svg')
            plt.close()


