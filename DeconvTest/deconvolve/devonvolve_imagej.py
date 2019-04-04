"""
Module containing functions running ImageJ deconvolution plugins with given parameters. 
Includes the following ImageJ plugins / algorithms:
- Iterative deconvolve 3D
- DeconvolutionLab2: Regularized Inverse Filter (RIF)
- DeconvolutionLab2: Richardson-Lucy with Total Variance (RLTV)
"""
from __future__ import division

import os


def run_rif(imagej_path, inputfile, psffile, rif_lambda, outputfile):
    """
    Runs Regularized Inverse Filter (RIF) from DeconvolutionLab2 plugin with given parameters.
    
    Parameters
    ----------
    imagej_path : str
        Absolute path to the Fiji / ImageJ software
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    rif_lambda : float
        Regularization parameter for the RIF algorithm.
    outputfile : str
        Absolute path to the output file.

    """

    os.system(imagej_path + " --headless --console -macro rif.ijm '" + inputfile + ' ' +
              psffile + ' ' + str(rif_lambda) + ' ' + outputfile + "'")


def run_rltv(imagej_path, inputfile, psffile, iterations, rltv_lambda, outputfile):
    """
    Runs Richardson-Lucy with Total Variance (RLTV) from DeconvolutionLab2 plugin with given parameters.
    
    Parameters
    ----------
    imagej_path : str
        Absolute path to the Fiji / ImageJ software
    inputfile : str
        Absolute path to the input file that should be deconvolved.
    psffile : str
        Absolute path to the psf file.
    iterations : int
        Number of iterations in the RLTV algorithm.
    rltv_lambda : float
        Regularization parameter for the RLTV algorithm.
    outputfile : str
        Absolute path to the output file.


    """

    os.system(imagej_path + " --headless --console -macro rltv.ijm '" + inputfile + ' ' +
              psffile + ' ' + str(iterations) + ' ' + str(rltv_lambda) + ' ' + outputfile + "'")


def run_iterative(imagej_path, inputfile, psffile, outputfile, normalize, perform, detect, wiener,
                  low, terminate, iterations):
    """   
    Runs Iterative Deconvolve 3D plugin with given parameters.
    
    Parameters
    ----------
    imagej_path : str
        Absolute path to the Fiji / ImageJ software
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

    """

    os.system(imagej_path + " --headless --console -macro iterative.ijm '" + inputfile + ' ' +
              psffile + ' ' + outputfile + ' ' + str(normalize).upper() + ' ' + str(perform).upper() +
              ' ' + str(detect).upper() + ' ' + str(wiener) + ' ' + str(low) + ' ' +
              str(terminate) + ' ' + str(iterations) + "'")






