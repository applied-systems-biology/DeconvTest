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

    pairs = [['Poisson_SNR', 'Algorithm'], ['Resolution', 'Algorithm'],
             ['PSF sigma X', 'Algorithm'], ['PSF elongation', 'Algorithm']]

    fn = accuracy_folder[:-1] + '.csv'
    stat = pd.read_csv(fn, sep='\t', index_col=0)
    stat = extract_metadata(stat)

    for pair in pairs:
        stat = stat.sort_values(pair)
        plot(stat, cols, plot_folder + 'average/', labels=labels, x=pair[0], hue=pair[1], figsize=(5, 4),
             rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})

    best_stat = choose_best_settings(stat, col='Jaccard index')
    best_stat.to_csv(fn[:-4] + '_best_settings.csv', sep='\t')
    plot_best_settings(best_stat, plot_folder + 'best_settings/')
    plot_details(stat, plot_folder, labels, cols)

    fn = log_folder[:-1] + '.csv'
    stat = pd.read_csv(fn, sep='\t', index_col=0)
    stat = extract_metadata(stat)
    for pair in pairs:
        stat = stat.sort_values(pair)
        plot(stat, cols, plot_folder + 'average/', labels=labels, x=pair[0], hue=pair[1], logscale=True,
             rotation=90, margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2}, figsize=(5, 4))


def choose_best_settings(stat, col=None):
    if col is None:
        col = 'Jaccard index'
    stat_summary = stat.groupby(['PSF sigma X', 'PSF elongation', 'Resolution',
                                 'Poisson_SNR', 'Algorithm', 'Settings']).mean().reset_index()
    best_stat = pd.DataFrame()
    for sigma in stat_summary['PSF sigma X'].unique():
        for elon in stat_summary['PSF elongation'].unique():
            for res in stat_summary['Resolution'].unique():
                for snr in stat_summary['Poisson_SNR'].unique():
                    for alg in stat_summary['Algorithm'].unique():
                        paramsstat = stat_summary[(stat_summary['PSF sigma X'] == sigma)
                                                  & (stat_summary['PSF elongation'] == elon)
                                                  & (stat_summary['Resolution'] == res)
                                                  & (stat_summary['Poisson_SNR'] == snr)
                                                  & (stat_summary['Algorithm'] == alg)]

                        paramsstat = paramsstat.sort_values(col, ascending=False).reset_index()
                        paramsstat.at[:, 'PSF size'] = 'XY=' + str(sigma) + '$\mu m$; aspect=' + str(elon)
                        paramsstat.at[:, 'Jaccard index variation minmax'] = paramsstat.iloc[0]['Jaccard index'] \
                                                                             - paramsstat.iloc[-1]['Jaccard index']
                        paramsstat.at[:, 'Jaccard index variation 1-2'] = paramsstat.iloc[0]['Jaccard index'] \
                                                                             - paramsstat.iloc[1]['Jaccard index']
                        best_stat = pd.concat([best_stat, paramsstat.iloc[0:1]])

    best_stat = best_stat.reset_index(drop=True)
    return best_stat


def plot_best_settings(stat, plot_folder):
    cols = ['Jaccard index variation minmax', 'Jaccard index variation 1-2']

    for resolution in stat['Resolution'].unique():
        paramsstat = stat[stat['Resolution'] == resolution]
        res = str(resolution).replace('[', '').replace(']', '').replace(' ', '_').replace('.', 'p').replace('__', '_')

        alg = 'DeconvolutionLab-RIF'
        curstat = paramsstat[paramsstat['Algorithm'] == alg]
        if len(curstat) > 0:
            plot(curstat, ['Lambda'], plot_folder + alg + '_resolution=' + res + '_',
                 x='PSF size', hue='Poisson_SNR', rotation=90, figsize=(4, 6), logscale=True,
                 title=alg, normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35})
            plot(curstat, cols, plot_folder + alg + '_resolution=' + res + '_',
                 x='PSF size', hue='Poisson_SNR', rotation=90, figsize=(4, 6),
                 title=alg, normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35})

        alg = 'DeconvolutionLab-RLTV'
        curstat = paramsstat[paramsstat['Algorithm'] == alg]
        if len(curstat) > 0:
            plot(curstat, ['Lambda'], plot_folder + alg + '_resolution=' + res + '_',
                 x='PSF size', hue='Poisson_SNR', rotation=90, figsize=(4, 6), logscale=True,
                 title=alg, normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35})
            plot(curstat, ['Iterations'] + cols, plot_folder + alg + '_resolution=' + res + '_',
                 x='PSF size', hue='Poisson_SNR', rotation=90, figsize=(4, 6),
                 title=alg, normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35})

        alg = 'IterativeDeconvolve3D'
        curstat = paramsstat[paramsstat['Algorithm'] == alg]
        if len(curstat) > 0:
            plot(curstat, ['Normalize PSF', 'Perform anti-ringing', 'Detect divergence', 'Low pass filter, pixels',
                           'Terminate if mean delta <', 'Wiener filter gamma'] + cols,
                 plot_folder + alg + '_resolution=' + res + '_',
                 x='PSF size', hue='Poisson_SNR', rotation=90, figsize=(4, 6),
                 title=alg, normalize=True, margins={'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.35})


def plot_details(stat, plot_folder, labels, cols):
    for sigma in stat['PSF sigma X'].unique():
        for elon in stat['PSF elongation'].unique():
            for res in stat['Resolution'].unique():
                for snr in stat['Poisson_SNR'].unique():
                    paramsstat = stat[(stat['PSF sigma X'] == sigma) & (stat['PSF elongation'] == elon)
                                      & (stat['Resolution'] == res) & (stat['Poisson_SNR'] == snr)]
                    current_plot_folder = plot_folder + 'detailed/sigma=' + str(sigma).replace('.', 'p') \
                                          + '_elongation=' + str(elon).replace('.', 'p') \
                                          + '_resolution=' + str(res).replace('[', '').replace(']', '')\
                                              .replace(' ', '_').replace('.', 'p').replace('__', '_') \
                                          + '_poisson_snr=' + str(snr) + '/'

                    alg = 'DeconvolutionLab-RIF'
                    curstat = paramsstat[paramsstat['Algorithm'] == alg]
                    if len(curstat) > 0:
                        plot(curstat, cols, current_plot_folder + alg + '_', labels=labels, x='Lambda', rotation=90,
                             figsize=(5, 4), title=alg, normalize=True,
                             margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})

                    alg = 'DeconvolutionLab-RLTV'
                    curstat = paramsstat[paramsstat['Algorithm'] == alg]
                    curstat = curstat[curstat['Lambda'] < 10]
                    if len(curstat) > 0:
                        plot(curstat, cols, current_plot_folder + alg + '_', x='Lambda', labels=labels,
                             hue='Iterations', rotation=90, title=alg, figsize=(5, 4), normalize=True,
                             margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})

                    alg = 'IterativeDeconvolve3D'
                    curstat = paramsstat[paramsstat['Algorithm'] == alg]
                    if len(curstat) > 0:
                        hues = ['Normalize PSF', 'Perform anti-ringing', 'Detect divergence']
                        xs = ['Low pass filter, pixels', 'Terminate if mean delta <', 'Wiener filter gamma']
                        for x in xs:
                            for hue in hues:
                                plot(curstat, cols, current_plot_folder + alg + '_', x=x,
                                     labels=labels, hue=hue, rotation=90, title=alg, figsize=(5, 4), normalize=True,
                                     margins={'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.2})


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















