import os
import sys
import shutil
import zipfile
import pandas as pd
from helper_lib import filelib


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
    os.system(get_fiji_path() + " --headless --console -macro version.ijm '" + outputfolder + "'")

