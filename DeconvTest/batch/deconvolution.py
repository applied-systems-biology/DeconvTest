"""
Module containing functions for deconvolving images in a batch mode.
"""
from __future__ import division

import time
import os
import numpy as np
import pandas as pd
import itertools

from DeconvTest.modules import deconvolution
from DeconvTest.classes.metadata import Metadata
from helper_lib.parallel import run_parallel
from helper_lib import filelib


def deconvolve_batch(inputfolder, outputfolder, deconvolution_algorithm, **kwargs):
    """
    Deconvolves all cell images in a given input directory with multiple algorithm and settings.
    
    Parameters
    ----------
    inputfolder : str
        Input directory with cell images to fiji.
    outputfolder : str
        Output directory to save the deconvolved images.
    deconvolution_algorithm : string, sequence of strings
        Name of the deconvolution algorithm from set of
         {deconvolution_lab_rif, deconvolution_lab_rltv, iterative_deconvolve_3d}.
        If a sequence is provided, all algorithms from the sequence will be tested.

    Keyword arguments
    -----------------
    <deconvolution_algorithm>_<parameter> : scalar or sequence
        Values of the parameters for the deconvolution algorithms to be tested.
        <deconvolution_algorithm> is the name of the algorithm from set of
         {deconvolution_lab_rif, deconvolution_lab_rltv, iterative_deconvolve_3d}
         for which the parameters values refer to.
        <parameter> is the name of the parameter of the specified algorithm.
        For instance, 'deconvolution_lab_rltv_iterations' specifies the value(s) for the number of iterations of the
         'deconvolution_lab_rltv' algorithm.
        If a sequence of parameter values is provided, all values from the sequence will be tested.
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
    algorithm = np.array([deconvolution_algorithm]).flatten()

    items = []
    for alg in algorithm:
        alg_params = []
        alg_param_names = []
        for kw in kwargs:
            if kw.startswith(alg):
                alg_param_names.append(kw)
                alg_params.append(np.array([kwargs[kw]]).flatten())
        alg_params = list(itertools.product(*alg_params))
        for cur_params in alg_params:
            param_args = dict()
            for i in range(len(alg_param_names)):
                param_args[alg_param_names[i].split(alg)[-1][1:]] = cur_params[i]
            items.append((alg, param_args))
    kwargs['items'] = [(inputfile,) + item for inputfile in inputfiles for item in items]
    kwargs['outputfolder'] = outputfolder
    kwargs['inputfolder'] = inputfolder
    kwargs['imagej_path'] = deconvolution.get_fiji_path()
    if deconvolution.get_fiji_path() is None:
        raise TypeError("Fiji path is not specified! Run the `python setup.py install` and specify the Fiji path")

    run_parallel(process=__deconvolve_batch_helper, process_name='Deconvolve', **kwargs)


def __deconvolve_batch_helper(item, inputfolder, outputfolder, imagej_path,
                              log_computing_time=True, logfolder=None, **kwargs_to_ignore):
    filename, algorithm, alg_kwargs = item
    name = filename.split('/')[-1]
    elapsed_time = None
    metadata = Metadata()
    metadata.read_from_csv(inputfolder + filename[:-4] + '.csv')
    if 'isPSF' not in metadata.index or str(metadata['isPSF']) == 'False':
        psfname = filename[: -len(name)-1].split('_noise')[0] + '.tif'
        subfolder = algorithm
        for kw in alg_kwargs:
            subfolder += '_' + kw + '=' + str(alg_kwargs[kw])
        subfolder += '/'
        if not os.path.exists(outputfolder + subfolder + filename):
            filelib.make_folders([outputfolder + subfolder + os.path.dirname(filename)])
            start = time.time()
            if algorithm in dir(deconvolution) and algorithm in deconvolution.valid_algorithms:
                getattr(deconvolution, algorithm)(imagej_path=imagej_path,
                                                  inputfile=os.getcwd() + '/' + inputfolder + filename,
                                                  psffile=os.getcwd() + '/' + inputfolder + psfname,
                                                  outputfile=os.getcwd() + '/' + outputfolder + subfolder + filename,
                                                  **alg_kwargs)
                elapsed_time = time.time() - start
            else:
                raise AttributeError(algorithm + ' is not a valid algorithm!')
            metadata = Metadata(filename=inputfolder + filename[:-4] + '.csv')
            metadata['Deconvolution algorithm'] = algorithm
            for c in alg_kwargs:
                metadata[c] = alg_kwargs[c]
            metadata.save(outputfolder + subfolder + filename[:-4] + '.csv')

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
            for c in metadata.index:
                try:
                    t[c] = metadata[c]
                except ValueError:
                    t[c] = str(metadata[c])
            t.to_csv(logfolder + subfolder[:-1] + '_' + filename[:-4].replace('/', '_') + '.csv', sep='\t')









