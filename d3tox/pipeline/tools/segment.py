import logging

import numpy as np

from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import erosion

from d3tox.pipeline.tools.centers import find_mask_center_of_mass


def _eucl_dist(origin_yx, coords):
    dist = coords - origin_yx
    dist = np.sqrt(np.sum(dist ** 2, axis=1))

    return dist


def _ensure_point_is_on_mask_edge(msk, corner):
    """
    Make sure that the corner point is on the masks edge.
    This prevents off-by-1-pixel errors caused by rounding errors on certain transformations / processing on the imaging.
    """
    edge = msk.copy()
    edge[erosion(edge) > 0] = 0
    edge[edge > 0] = 1

    edge_yx = np.where(edge > 0)
    edge_yx = np.column_stack(edge_yx)

    dist = _eucl_dist(corner, edge_yx)
    nn_yx = edge_yx[(np.abs(dist)).argsort()[0]]

    return nn_yx


def segment_discs(vertebrae_corners_xy, segmentations, vertebrae_angles):
    discs = [f'C{x}' for x in range(2, 8)]

    masks = {}
    angles_guess = {}
    centers = {}

    for left, right in zip(discs[1:], discs[:-1]):
        if left not in vertebrae_corners_xy or right not in vertebrae_corners_xy:
            logging.info(f'Skipping disc {left}-{right} as one or more vertebrae are missing')
            continue

        disc_name = f'{left}-{right}'

        angle_guess = (vertebrae_angles[left] + vertebrae_angles[right]) / 2

        msk_left = segmentations[left]
        msk_right = segmentations[right]

        msk = np.zeros(msk_left.shape, dtype=np.uint8)
        msk[msk_left > 0] = 255
        msk[msk_right > 0] = 255

        pimg = Image.fromarray(msk).convert('L')
        draw = ImageDraw.Draw(pimg)

        vertebrae_corners_xy[left]['top_right'] = _ensure_point_is_on_mask_edge(msk_left, vertebrae_corners_xy[left]['top_right'][::-1])
        vertebrae_corners_xy[left]['bottom_right'] = _ensure_point_is_on_mask_edge(msk_left, vertebrae_corners_xy[left]['bottom_right'][::-1])

        vertebrae_corners_xy[right]['top_left'] = _ensure_point_is_on_mask_edge(msk_right, vertebrae_corners_xy[right]['top_left'][::-1])
        vertebrae_corners_xy[right]['bottom_left'] = _ensure_point_is_on_mask_edge(msk_right, vertebrae_corners_xy[right]['bottom_left'][::-1])

        p1 = vertebrae_corners_xy[left]['top_right'][::-1]
        p2 = vertebrae_corners_xy[right]['top_left'][::-1]
        draw.line((*p1, *p2), fill=255, width=1)

        p1 = vertebrae_corners_xy[left]['bottom_right'][::-1]
        p2 = vertebrae_corners_xy[right]['bottom_left'][::-1]
        draw.line((*p1, *p2), fill=255, width=1)

        img_disc = np.asarray(pimg.convert('L'))

        _debug_imsave(img_disc, f'full-{disc_name}')

        img_disc = ndimage.binary_fill_holes(img_disc).astype(np.uint8)
        img_disc[np.where(img_disc >= 1)] = 255

        img_disc[msk_left > 0] = 0
        img_disc[msk_right > 0] = 0

        # select the largest blob as the disc
        labels, nlabels = ndimage.label(img_disc > 1)
        size = np.bincount(labels.ravel())
        biggest_label = size[1:].argmax() + 1
        clump_mask = labels == biggest_label

        img_disc[np.logical_not(clump_mask)] = 0
        img_disc[clump_mask] == 255

        masks[disc_name] = img_disc
        angles_guess[disc_name] = np.deg2rad(angle_guess)
        centers[disc_name] = find_mask_center_of_mass(img_disc)

        _debug_imsave(img_disc, f'seg-{disc_name}')

        # #img_lines = np.asarray(pimg.cget_nii_file_ndim_shapeonvert('L'))

    return masks, angles_guess, centers

def _debug_imsave(img, name):
    return

    pimg = Image.fromarray(img).convert('L')
    pimg.save(f'/home/thomas/tmp/cervout/test2/{name}.png')