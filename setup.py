import sys
import os
import shutil
import pandas as pd


def check_imagej_path(path):
    name = path.split('/')[-1]
    if len(name.split('ImageJ')) > 1:
        path = os.path.dirname(path)
    if not path.endswith('/'):
        path += '/'
    if not os.path.exists(path):
        return None
    files = os.listdir(path)
    for fn in files:
        # if len(fn.split('ImageJ')) > 1 and not fn.endswith('desktop'):
        if len(fn.split('ImageJ')) > 1:
            return path + fn
    return None


if __name__ == '__main__':

    args = sys.argv[1:]
    imagej_path = None
    if os.path.exists('DeconvTest/config'):
        stat = pd.read_csv('DeconvTest/config', sep='\t', index_col=0, header=-1).transpose().iloc[0].T
        if 'Fiji_path' in stat.index:
            imagej_path = stat['Fiji_path']
        else:
            imagej_path = None
    if imagej_path is None:
        imagej_dir = raw_input('Enter the Fiji path: ')
        imagej_path = check_imagej_path(imagej_dir)
        while imagej_path is None:
            imagej_dir = raw_input('The inserted path does not exist. Enter the Fiji path: ')
            imagej_path = check_imagej_path(imagej_dir)

        config = pd.Series({'Fiji_path': imagej_path})
        config.to_csv('DeconvTest/config', sep='\t')

    if not os.path.exists(os.path.dirname(imagej_path) + '/macros'):
        os.makedirs(os.path.dirname(imagej_path) + '/macros')

    shutil.copy('DeconvTest/deconvolve/version.ijm', os.path.dirname(imagej_path) + '/macros/' + 'version.ijm')
    shutil.copy('DeconvTest/deconvolve/rif.ijm', os.path.dirname(imagej_path) + '/macros/' + 'rif.ijm')
    shutil.copy('DeconvTest/deconvolve/rltv.ijm', os.path.dirname(imagej_path) + '/macros/' + 'rltv.ijm')
    shutil.copy('DeconvTest/deconvolve/iterative.ijm', os.path.dirname(imagej_path) + '/macros/' + 'iterative.ijm')

    command = 'python ./.setup.py '
    for arg in args:
        command += arg + ' '
    os.system(command)

