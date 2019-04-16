"""
Module containing functions for deconvolving images with ImageJ plugins in a batch mode.
Includes the following ImageJ plugins / algorithms:
- Iterative fiji 3D
- DeconvolutionLab2: Regularized Inverse Filter (RIF)
- DeconvolutionLab2: Richardson-Lucy with Total Variance (RLTV)
"""
from __future__ import division

import time
import numpy as np

from DeconvTest.modules.deconvolution_fiji import *
from DeconvTest.classes.metadata import Metadata
from helper_lib.parallel import run_parallel
from helper_lib import filelib


def deconvolve_batch(inputfolder, outputfolder, algorithm, rif_lambda=0.01, rltv_lambda=0.01, iterations=20,
                     normalize=True, perform=True, detect=True, wiener=0.001, low=1,
                     terminate=0.001, log_computing_time=True, logfolder=None, print_progress=False, max_threads=8):
    """
    Deconvolves all cell images in a given input directory with multiple algorithm and settings.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to fiji.
    outputfolder : str
        Output directory to save the deconvolved images.
    algorithm : str or sequence of str
        Deconvolution algorithm that will be applied to fiji the input images.
        If a sequence of algorithms is provided, all algorithms from the sequence will be applied.
        The valid values are as follows:
        
        'deconvolution_lab_rif'
            Regularized Inverse Filter (RIF) from DeconvolutionLab2 plugin (ImageJ) will be applied.
        
        'deconvolution_lab_rltv'
            Richardson-Lucy with Total Variance (RLTV) from DeconvolutionLab2 plugin (ImageJ) will be applied.
        
        'iterative_deconvolve_3d'
            Iterative Deconvolve 3D plugin (ImageJ) will be applied.
    rif_lambda : float or sequence of floats, optional
        Regularization parameter for the RIF algorithm.
        If a sequence is provided, all values from the sequence will be tested.
        Default is 0.01.
    rltv_lambda : float or sequence of floats, optional
        Regularization parameter for the RLTV algorithm.
        If a sequence is provided, all values from the sequence will be tested in combination with 
         the `iterations` parameter.
        Default is 0.01.
    iterations : int or sequence of ints, optional
        Number of iterations in the RLTV algorithm.
        If a sequence is provided, all values from the sequence will be tested in combination with 
         the `rltv_lambda` parameter.
        Default is 20.
    normalize : bool or sequence of bools, optional
        `Normalize PSF` parameter for the Iterative Deconvolve 3D plugin. 
        If a sequence is provided, all values from the sequence will be tested in combination with other parameters.
        Default is True.
    perform : bool or sequence of bools, optional
        `Perform anti-ringig step` parameter for the Iterative Deconvolve 3D plugin. 
        If a sequence is provided, all values from the sequence will be tested in combination with other parameters.
        Default is True.
    detect : bool or sequence of bools, optional
        `Detect divergence` parameter for the Iterative Deconvolve 3D plugin. 
        If a sequence is provided, all values from the sequence will be tested in combination with other parameters.
        Default is True.
    wiener : float or sequence of floats, optional
        `Wiener filter gamma` parameter for the Iterative Deconvolve 3D plugin.
        <0.0001 to turn off, 0.0001 - 0.1 as tests. 
        If a sequence is provided, all values from the sequence will be tested in combination with other parameters.
        Default is 0.001.
    low : int or sequence of ints, optional
        `Low pass filter` parameter for the Iterative Deconvolve 3D plugin, pixels.
        The same value is used for low pass filter in xy and z.
        If a sequence is provided, all values from the sequence will be tested in combination with other parameters.
        Default is 1.
    terminate : float or sequence of floats, optional
        `Terminate iteration if mean delta < x%` parameter for the Iterative Deconvolve 3D plugin.
        0 to turn off.
        If a sequence is provided, all values from the sequence will be tested in combination with other parameters.
        Default is 0.001.
    log_computing_time : bool, optional
        If True, computing time spent on deconvolution will be recorded and stored in a given folder.
        Default is False.
    logfolder : str, optional
        Directory to store computing time when `log_computing_time` is set to True.
        If None, the logfolder will be set to `outputfolder` + "../log/".
        Default is None.
    max_threads : int, optional
        The maximal number of processes to run in parallel.
        Default is 8.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.

    """
    if not inputfolder.endswith('/'):
        inputfolder += '/'
    if not outputfolder.endswith('/'):
        outputfolder += '/'
    inputfiles = filelib.list_subfolders(inputfolder)
    algorithm = np.array([algorithm]).flatten()
    rif_lambda = np.array([rif_lambda]).flatten()
    rltv_lambda = np.array([rltv_lambda]).flatten()
    iterations = np.array([iterations]).flatten()
    normalize = np.array([normalize]).flatten()
    perform = np.array([perform]).flatten()
    detect = np.array([detect]).flatten()
    wiener = np.array([wiener]).flatten()
    low = np.array([low]).flatten()
    terminate = np.array([terminate]).flatten()

    for alg in algorithm:
        if alg not in ['deconvolution_lab_rif', 'deconvolution_lab_rltv', 'iterative_deconvolve_3d']:
            raise ValueError(alg + ' is invalid value for the algorithm! '
                             'Must be \'deconvolution_lab_rif\', \'deconvolution_lab_rltv\','
                             ' \'iterative_deconvolve_3d\', or a list these')

    if 'iterative_deconvolve_3d' in algorithm:
        items = [(inputfile, 'iterative_deconvolve_3d', norm, perf, det, wien, lo, term)
                      for inputfile in inputfiles for norm in normalize for perf in perform for det in detect
                      for wien in wiener for lo in low for term in terminate]
    else:
        items = []
    if 'deconvolution_lab_rif' in algorithm:
        for inputfile in inputfiles:
            for rl in rif_lambda:
                items.append((inputfile, 'deconvolution_lab_rif', rl))
    if 'deconvolution_lab_rltv' in algorithm:
        for inputfile in inputfiles:
            for rl in rltv_lambda:
                for it in iterations:
                    items.append((inputfile, 'deconvolution_lab_rltv', rl, it))

    # filelib.make_folders([outputfolder])
    # f = open(outputfolder + 'items.txt', 'w')
    # f.write(str(items))
    # f.close()

    imagej_path = get_fiji_path()
    if imagej_path is None:
        raise TypeError("Fiji path is not specified! Run the `python setup.py install` and specify the Fiji path")

    kwargs = {'items': items, 'inputfolder': inputfolder, 'outputfolder': outputfolder,
              'log_computing_time': log_computing_time, 'logfolder': logfolder, 'imagej_path': imagej_path,
              'max_threads': max_threads, 'print_progress': print_progress}
    run_parallel(process=__deconvolve_batch_helper, process_name='Deconvolve', **kwargs)


def __deconvolve_batch_helper(item, inputfolder, outputfolder, imagej_path, log_computing_time=True, logfolder=None):
    filename, algorithm = item[:2]
    name = filename.split('/')[-1]
    elapsed_time = None
    if len(name.split('psf')) == 1:
        psfname = filename[: -len(name)-1].split('_gaussian')[0] + '.tif'
        if algorithm == 'deconvolution_lab_rif':
            rif_lambda = item[2]
            subfolder = 'DeconvolutionLab-RIF_lambda=' + str(rif_lambda) + '/'
            if not os.path.exists(outputfolder + subfolder + filename):
                filelib.make_folders([outputfolder + subfolder + os.path.dirname(filename)])
                start = time.time()
                run_rif(imagej_path=imagej_path, inputfile=os.getcwd() + '/' + inputfolder + filename,
                        psffile=os.getcwd() + '/' + inputfolder + psfname, rif_lambda=rif_lambda,
                        outputfile=os.getcwd() + '/' + outputfolder + subfolder + filename)
                elapsed_time = time.time() - start
        if algorithm == 'deconvolution_lab_rltv':
            rltv_lambda, iterations = item[2:4]
            subfolder = 'DeconvolutionLab-RLTV_lambda=' + str(rltv_lambda) + '_iterations=' + str(iterations) + '/'
            if not os.path.exists(outputfolder + subfolder + filename):
                filelib.make_folders([outputfolder + subfolder + os.path.dirname(filename)])
                start = time.time()
                run_rltv(imagej_path=imagej_path, inputfile=os.getcwd() + '/' + inputfolder + filename,
                         psffile=os.getcwd() + '/' + inputfolder + psfname, iterations=iterations, rltv_lambda=rltv_lambda,
                         outputfile=os.getcwd() + '/' + outputfolder + subfolder + filename)
                elapsed_time = time.time() - start
        if algorithm == 'iterative_deconvolve_3d':
            normalize, perform, detect, wiener, low, terminate = item[2:]
            subfolder = 'IterativeDeconvolve3D_normalize=' + str(normalize) + '_perform=' + str(perform) + \
                            '_detect=' + str(detect) + '_wiener=' + str(wiener) + '_low=' + str(low) + \
                            '_terminate=' + str(terminate) + '/'
            if not os.path.exists(outputfolder + subfolder + filename):
                filelib.make_folders([outputfolder + subfolder + os.path.dirname(filename)])
                start = time.time()
                run_iterative(imagej_path=imagej_path, inputfile=os.getcwd() + '/' + inputfolder + filename,
                              psffile=os.getcwd() + '/' + inputfolder + psfname,
                              outputfile=os.getcwd() + '/' + outputfolder + subfolder + filename,
                              normalize=normalize, perform=perform, detect=detect, wiener=wiener,
                              low=low, terminate=terminate, iterations=200)
                elapsed_time = time.time() - start

        if log_computing_time is True and elapsed_time is not None:
            if logfolder is None:
                logfolder = outputfolder + '../log/'
            else:
                if not logfolder.endswith('/'):
                    logfolder += '/'

            filelib.make_folders([logfolder])
            t = pd.DataFrame({'Step': ['Deconvolution'],
                              'Computational time': [elapsed_time],
                              'Algorithm': algorithm,
                              'Name': subfolder[:-1] + '/' + filename})
            t.to_csv(logfolder + subfolder[:-1] + '_' + filename[:-4].replace('/', '_') + '.csv', sep='\t')
        metadata = Metadata(filename=inputfolder + filename[:-4] + '.csv')
        metadata.save(outputfolder + subfolder + filename[:-4] + '.csv')










