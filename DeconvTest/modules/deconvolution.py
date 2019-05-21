"""
This module contains functions for running Fiji deconvolution plugins with given parameters.
"""
from __future__ import division
import os
import sys
import shutil
import zipfile
import pandas as pd
from helper_lib import filelib

valid_algorithms = ['deconvolution_lab_rif', 'deconvolution_lab_rltv', 'iterative_deconvolve_3d']


def deconvolution_lab_rif(inputfile, psffile, outputfile, regularization_lambda=0.01,
                          imagej_path=None, **kwargs_to_ignore):
    """
    Runs Regularized Inverse Filter (RIF) from DeconvolutionLab2 plugin with given parameters.
    
    Parameters
    ----------
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    outputfile : str
        Absolute path to the output file.
    regularization_lambda : float, optional
        Regularization parameter for the RIF algorithm.
        Default is 0.01
    imagej_path : str
        Absolute path to the Fiji / ImageJ software.

    """
    if imagej_path is None:
        imagej_path = get_fiji_path()

    os.system(imagej_path + " --headless --console -macro deconvolution_lab_rif.ijm '" + inputfile + ' ' +
              psffile + ' ' + str(regularization_lambda) + ' ' + os.path.dirname(outputfile) + ' ' +
              os.path.basename(outputfile)[:-4] + "'")


def deconvolution_lab_rltv(inputfile, psffile, outputfile, iterations=10, regularization_lambda=0.01,
                           imagej_path=None, **kwargs_to_ignore):
    """
    Runs Richardson-Lucy with Total Variance (RLTV) from DeconvolutionLab2 plugin with given parameters.
    
    Parameters
    ----------
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    outputfile : str
        Absolute path to the output file.
    iterations : int, optional
        Number of iterations in the RLTV algorithm.
        Default is 10.
    regularization_lambda : float, optional
        Regularization parameter for the RLTV algorithm.
        Default is 0.01
    imagej_path : str
        Absolute path to the Fiji / ImageJ software

    """
    if imagej_path is None:
        imagej_path = get_fiji_path()

    os.system(imagej_path + " --headless --console -macro deconvolution_lab_rltv.ijm '" + inputfile + ' ' +
              psffile + ' ' + str(iterations) + ' ' + str(regularization_lambda) + ' ' +
              os.path.dirname(outputfile) + ' ' + os.path.basename(outputfile)[:-4] + "'")


def iterative_deconvolve_3d(inputfile, psffile, outputfile, normalize=False, perform=True,
                            detect=False, wiener=0.5, low=1, terminate=0.001,
                            iterations=200, imagej_path=None, **kwargs_to_ignore):
    """
    Runs Iterative Deconvolve 3D plugin with given parameters.
    
    Parameters
    ----------
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    outputfile : str
        Absolute path to the output file.
    normalize : bool, optional
        `Normalize PSF` parameter for the Iterative Deconvolve 3D plugin.
        Normalizes the PSF before deconvoltution.
        Default is False.
    perform : bool, optional
        `Perform anti-ringing step` parameter for the Iterative Deconvolve 3D plugin.
        Reduces artifacts from features close to the image border.
        Default is True.
    detect : bool, optional
        `Detect divergence` parameter for the Iterative Deconvolve 3D plugin.
        Stops the iteration if the changes in the image increase.
        Default is False.
    wiener : float, optional
        `Wiener filter gamma` parameter for the Iterative Deconvolve 3D plugin.
        <0.0001 to turn off, 0.0001 - 0.1 as tests.
        Default is 0.5.
    low : int, optional
        `Low pass filter` parameter for the Iterative Deconvolve 3D plugin, pixels.
        The same value is used for low pass filter in xy and z.
        Default is 1.
    terminate : float, optional
        `Terminate iteration if mean delta < x%` parameter for the Iterative Deconvolve 3D plugin.
        Stops the iteration if the changes in the image are less than the value of x.
        0 to turn off.
        Default is 0.001.
    iterations : int, optional
        Number of iterations in the Iterative Deconvolve 3D algorithm.
        Default is 200.
    imagej_path : str
        Absolute path to the Fiji / ImageJ software

    """

    if imagej_path is None:
        imagej_path = get_fiji_path()

    os.system(imagej_path + " --headless --console -macro iterative_deconvolve_3d.ijm '" + inputfile + ' ' +
              psffile + ' ' + outputfile + ' ' + str(normalize).upper() + ' ' + str(perform).upper() +
              ' ' + str(detect).upper() + ' ' + str(wiener) + ' ' + str(low) + ' ' +
              str(terminate) + ' ' + str(iterations) + "'")


########################################################

def get_fiji_path():
    """
    Returns the Fiji path as specified by a config file.

    Returns
    -------
    str
        Absolute path to the Fiji software.

    """
    imagej_path = None
    for path in sys.path:
        if os.path.exists(path) and os.path.isdir(path):
            files = os.listdir(path)
            for fn in files:
                if len(fn.split('deconvtest')) > 1:
                    if zipfile.is_zipfile(path + '/' + fn):
                        zf = zipfile.ZipFile(path + '/' + fn, 'r')
                        zf.extract('DeconvTest/fiji_path', path='temp/')
                        imagej_path = pd.read_csv('temp/DeconvTest/fiji_path', sep='\t',
                                                  index_col=0, header=-1).transpose().iloc[0].T['Fiji_path']
                        shutil.rmtree('temp')
                if imagej_path is not None:
                    break
    return imagej_path


def save_fiji_version(outputfolder):
    """
    Save the version of the Fiji software into a `version.txt` file in a give directory.

    Parameters
    ----------
    outputfolder : str
        Directory to store the Fiji version.

    """
    filelib.make_folders([outputfolder])
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    os.system(get_fiji_path() + " --headless --console -macro get_fiji_version.ijm '" + outputfolder + "'")




