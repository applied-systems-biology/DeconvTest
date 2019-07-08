"""
This module contains functions for quantification of deconvolution accuracy
"""

import pandas as pd
import numpy as np

from helper_lib.image import unify_shape


def compute_accuracy_measures(image, gt_image):
    """
    Computes accuracy measures between the current image and a given ground truth image.

    Parameters
    ----------
    image : ndarray
       Image to evaluate.
    gt_image : ndarray
        Ground truth image.

    Returns
    -------
    pandas.DataFrame()
        Data frame containing the values for the computed accuracy measures.
    """
    img_volume = np.product(gt_image.shape)
    image, gt_image = unify_shape(image, gt_image)  # convert cell images to the same shape
    data = pd.DataFrame()
    data['RMSE'] = [np.sqrt(np.sum((image - gt_image)**2) / img_volume)]
    data['NRMSE'] = data['RMSE'] / np.max(gt_image) - np.min(gt_image)
    return data





