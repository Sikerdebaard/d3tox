import logging

import nibabel as nib
import numpy as np

from scipy.ndimage import center_of_mass

from skimage.morphology import erosion, dilation
from skimage.measure import label, regionprops


def save_nii_take_headers_from(img_data, take_headers_from_this_nii_path, output_path):
    base_nii = load_nii(take_headers_from_this_nii_path)

    new_img = nib.Nifti1Image(img_data, base_nii.affine, base_nii.header)
    nib.save(new_img, output_path)


def load_nii(path):
    logging.debug(f'Loading nii {path}')
    nii = nib.load(path)

    return nii


def get_nii_file_ndim_shape(path, safe=True):
    nii = load_nii(path)

    if not safe:
        return nii.ndim, nii.shape

    data = get_nii_img(nii)

    return data.ndim, data.shape


def get_nii_mask_center_of_mass(msk):
    return [tuple(center_of_mass(msk))]


def _ensure_msk(nii):
    msk = get_nii_img(nii)
    msk[msk > 0] = 1
    return msk.astype(np.uint8)

def count_nii_mask_blobs(nii):
    msk = _ensure_msk(nii)

    # remove small artefacts by eroding and then dillating the image
    iters = 3
    for i in range(iters):
        msk = erosion(msk)

    for i in range(iters):
        msk = dilation(msk)

    labels_mask, numregions = label(msk, return_num=True)

    return numregions


def get_nii_mask_largest_blob(nii):
    msk = _ensure_msk(nii)

    # remove small artefacts by eroding and then dillating the image
    iters = 3
    for i in range(iters):
        msk = erosion(msk)

    for i in range(iters):
        msk = dilation(msk)

    # only keep the largest blob
    labels_mask = label(msk)
    regions = regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    # msk = labels_mask

    msk = np.zeros(msk.shape, np.uint8)
    msk[labels_mask != 0] = 1

    return msk


def ensure_grayscale(data):
    data = data.squeeze()  # remove axis of length 1

    if data.ndim > 2:
        if data.shape[-1] == 3:
            logging.debug(f'Converting RGB -> gray {data.shape}, {data.ndim}')
            # most likely RGB -> convert to gray
            return ensure_grayscale(np.dot(data[..., :], [0.2989, 0.5870, 0.1140]))

    return data


def get_nii_img(nii):
    data = nii.get_fdata()
    data = ensure_grayscale(data)

    return data
