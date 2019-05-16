"""
This module contains functions for quantification of deconvolution accuracy
"""

import pandas as pd
import numpy as np

from helper_lib.image import segment as sgm
from helper_lib.image import unify_shape


def segment(image, preprocess=False, thr=None, relative_thr=False, postprocess=False, label=True):
    """
    Segments the current image by thresholding with optional preprocessing and finding connected regions.

    Parameters
    ----------
    image : ndarray
        Input image
    preprocess : bool, optional
        If True, the image will be preprocessed with a median filter (size 3) prior to segmentation.
        Default is False.
    thr : scalar, optional
        Threshold value for image segmentation.
        If None, automatic Otsu threshold will be computed.
        Default is None.
    relative_thr : bool, optional
        If True, the value of `thr` is multiplied by the maximum intensity of the image.
        Default is False.
    postprocess bool, optional
        If True, morphological opening and closing and binary holes filling will be applied after theresholding.
        Default is False.
    label : bool, optional
        If True, connected region will be labeled by unique labels.
        Default is True.

    Returns
    -------
    ndarray
        Segmented binary mask or labeled image.
    """
    if image is None:
        raise ValueError('Image is None!')

    if preprocess:
        median = 3
    else:
        median = None
    if postprocess:
        morphology = True
        fill_holes = True
    else:
        morphology = False
        fill_holes = False

    image = sgm(image, thr=thr, relative_thr=relative_thr, median=median,
                morphology=morphology, fill_holes=fill_holes, label=label)

    if label is False:
        image = (image > 0) * 255

    return image


def compute_binary_accuracy_measures(image, gt_image):
    """
    Computes binary accuracy measures between the current image and a given ground truth image.

    Parameters
    ----------
    image : ndarray
        Binary image to evaluate.
    gt : ndarray
        Binary ground truth image.

    Returns
    -------
    pandas.DataFrame()
        Data frame containing the values for the computed accuracy measures.
    """
    image, gt_image = unify_shape(image, gt_image)  # convert cell images to the same shape
    image = (image > 0) * 1.
    gt_image = (gt_image > 0) * 1.
    overlap = np.sum(image * gt_image)
    union = np.sum((image + gt_image) > 0)
    a = np.sum(gt_image)
    b = np.sum(image)
    data = pd.DataFrame({'Jaccard index': [overlap / union],
                         'Sensitivity': overlap / a,
                         'Precision': overlap / b,
                         'Overdetection error': (b - overlap) / a,
                         'Underdetection error': (a - overlap) / a,
                         'Overlap error': (union - overlap) / a})
    return data


def compute_accuracy_measures(image, gt_image):
    """
    Computes accuracy measures between the current image and a given ground truth image.

    Parameters
    ----------
    image : ndarray
       Image to evaluate.
    gt : ndarray
        Ground truth image.

    Returns
    -------
    pandas.DataFrame()
        Data frame containing the values for the computed accuracy measures.
    """
    image, gt_image = unify_shape(image, gt_image)  # convert cell images to the same shape
    data = pd.DataFrame()
    data['RMSE'] = [np.sqrt(np.sum((image - gt_image)**2) / (np.product(image.shape)))]
    data['NRMSE'] = data['RMSE'] / np.max(gt_image) - np.min(gt_image)
    return data





