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


def deconvolution_lab_rif(inputfile, psffile, regularization_lambda, outputfile, imagej_path=None, **kwargs_to_ignore):
    """
    Runs Regularized Inverse Filter (RIF) from DeconvolutionLab2 plugin with given parameters.
    
    Parameters
    ----------
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    regularization_lambda : float
        Regularization parameter for the RIF algorithm.
    outputfile : str
        Absolute path to the output file.
    imagej_path : str
        Absolute path to the Fiji / ImageJ software

    """
    if imagej_path is None:
        imagej_path = get_fiji_path()

    os.system(imagej_path + " --headless --console -macro deconvolution_lab_rif.ijm '" + inputfile + ' ' +
              psffile + ' ' + str(regularization_lambda) + ' ' + os.path.dirname(outputfile) + ' ' +
              os.path.basename(outputfile)[:-4] + "'")


def deconvolution_lab_rltv(inputfile, psffile, iterations, regularization_lambda, outputfile,
                           imagej_path=None, **kwargs_to_ignore):
    """
    Runs Richardson-Lucy with Total Variance (RLTV) from DeconvolutionLab2 plugin with given parameters.
    
    Parameters
    ----------
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    iterations : int
        Number of iterations in the RLTV algorithm.
    regularization_lambda : float
        Regularization parameter for the RLTV algorithm.
    outputfile : str
        Absolute path to the output file.
    imagej_path : str
        Absolute path to the Fiji / ImageJ software

    """
    if imagej_path is None:
        imagej_path = get_fiji_path()

    os.system(imagej_path + " --headless --console -macro deconvolution_lab_rltv.ijm '" + inputfile + ' ' +
              psffile + ' ' + str(iterations) + ' ' + str(regularization_lambda) + ' ' +
              os.path.dirname(outputfile) + ' ' + os.path.basename(outputfile)[:-4] + "'")


def iterative_deconvolve_3d(inputfile, psffile, outputfile, normalize, perform, detect, wiener, low, terminate,
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
    normalize : bool
        `Normalize PSF` parameter for the Iterative Deconvolve 3D plugin. 
    perform : bool
        `Perform anti-ringig step` parameter for the Iterative Deconvolve 3D plugin. 
    detect : bool
        `Detect divergence` parameter for the Iterative Deconvolve 3D plugin. 
    wiener : float
        `Wiener filter gamma` parameter for the Iterative Deconvolve 3D plugin.
        <0.0001 to turn off, 0.0001 - 0.1 as tests. 
    low : int
        `Low pass filter` parameter for the Iterative Deconvolve 3D plugin, pixels.
        The same value is used for low pass filter in xy and z.
    terminate : float
        `Terminate iteration if mean delta < x%` parameter for the Iterative Deconvolve 3D plugin.
        0 to turn off.
    iterations : int
        Number of iterations in the Iterative Deconvolve 3D algorithm.
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




