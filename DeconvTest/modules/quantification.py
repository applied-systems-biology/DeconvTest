"""
This module contains functions for quantification of deconvolution accuracy
"""

from helper_lib.image import segment as sgm


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