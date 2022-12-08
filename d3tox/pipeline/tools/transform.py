from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, gaussian_filter
from skimage.morphology import erosion, dilation

import numpy as np

import d3tox


def rotate(img, angle, center, order):
    """
    Rotation by angle in counter-clockwise direction

    img:
        numpy array with image data

    angle:
        the rotation angle in degrees

    center:
        centerpoint of rotation

    order:
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    """

    center = np.round(center, 0).astype(int)

    pad_x = [img.shape[1] - center[0], center[0]]
    pad_y = [img.shape[0] - center[1], center[1]]
    img_p = np.pad(img, [pad_y, pad_x], 'constant')

    img_r = nd_rotate(img_p, -angle, reshape=False, order=order, mode='constant', cval=0)

    img_c = img_r[pad_y[0]: -pad_y[1], pad_x[0]: -pad_x[1]]

    return img_c


def move_centerpoint_to_center(img, centerpoint_xy, order):
    """
        Shift the image so that the new center becomes the centerpoint.

        img:
            numpy array with image data

        centerpoint_xy:
            the new centerpoint

        order:
            0: Nearest-neighbor
            1: Bi-linear (default)
            2: Bi-quadratic
            3: Bi-cubic
            4: Bi-quartic
            5: Bi-quintic
        """
    img_center = img.shape / 2
    shift = img_center - np.flip(centerpoint_xy)
    return nd_shift(img, shift, order=order)


def calc_crop_slicer_from_mask(msk, center, perc_extra=.1, min_extra=10):
    yidx, xidx = np.where(msk > 0)

    xmi, xma = xidx.min(), xidx.max()
    ymi, yma = yidx.min(), yidx.max()

    xlen = xma - xmi
    ylen = yma - ymi

    extra = np.max([xlen * perc_extra, ylen * perc_extra])
    if extra < min_extra:
        extra = min_extra

    xextra = extra
    yextra = extra

    left, right = np.round([xmi - xextra, xma + xextra], 0).astype(int)
    top, bottom = np.round([ymi - yextra, yma + yextra], 0).astype(int)

    top = top if top >= 0 else 0
    left = left if left >= 0 else 0
    right = right if right < msk.shape[1] else msk.shape[1] - 1
    bottom = bottom if bottom < msk.shape[0] else msk.shape[0] - 1

    new_center_xy = center[0] - left, center[1] - top

    return slice(top, bottom), slice(left, right), new_center_xy


def _score_overlap(msk):
    return len(np.where(msk <= 0)[0])


def _score_iou_inverse(msk):
    r_msk = msk.copy()
    r_msk[r_msk > 0] = 255
    b_msk = r_msk.copy()
    b_msk[b_msk >= 0] = 255

    intersection = np.logical_and(r_msk, b_msk)
    union = np.logical_or(r_msk, b_msk)
    iou_score = np.sum(intersection) / np.sum(union)

    # inverse
    return 1 - iou_score


def _score_xstrip(msk, length=2):
    msk = msk[:, -length:]

    return _score_overlap(msk)


def rotate_and_crop_to_msk(img, msk, angle, center, name, extra_opti=True, score_func=_score_overlap):
    angle_deg = np.degrees(angle)
    negation_angle = -angle_deg

    #is_c2 = False
    if name == 'C2':
        #is_c2 = True
        # we get slightly better results on C2 with _score_xstrip
        score_func = _score_xstrip

    if extra_opti:
        msk_pre = rotate(msk.copy(), angle_deg, center, 0)

        # opti_angle is in degrees
        opti_angle = find_optimal_box_rot(msk_pre, center.copy(), score_func)
        img = rotate(img, angle_deg + opti_angle, center, 3)
        msk = rotate(msk, angle_deg + opti_angle, center, 0)
        negation_angle = -(angle_deg + opti_angle)

        msk[msk > 0] = 255

        iters = 5
        for i in range(iters):
            msk = dilation(msk)

        for i in range(iters):
            msk = erosion(msk)

        msk = gaussian_filter(msk, sigma=1)
        thresh = 64
        msk[msk < thresh] = 0
        msk[msk >= thresh] = 255
    else:
        img = rotate(img, angle_deg, center, 3)
        msk = rotate(msk, angle_deg, center, 0)

    slicer_y, slicer_x, new_center = calc_crop_slicer_from_mask(msk, center)

    img = img[slicer_y, slicer_x]
    msk = msk[slicer_y, slicer_x]

    angle_deg_out = opti_angle if extra_opti else angle_deg

    return img, msk, new_center, negation_angle, angle_deg_out


def _rotate_and_score(msk, angle, center, score_func):
    r_msk = rotate(msk, angle, center, 0)

    yidx, xidx = np.where(r_msk > 0)
    xmi, xma = xidx.min(), xidx.max()
    ymi, yma = yidx.min(), yidx.max()

    r_msk = r_msk[ymi:yma, xmi:xma]

    score = score_func(r_msk)

    return score


def find_optimal_box_rot(msk, center, score_func):
    vals = {}

    yidx, xidx = np.where(msk > 0)
    xmi, xma = xidx.min(), xidx.max()
    ymi, yma = yidx.min(), yidx.max()

    msk_padded = msk[ymi:yma, xmi:xma]

    pad_len = max([xma - xmi, yma - ymi])
    msk_padded = np.pad(msk_padded, [[pad_len, pad_len], [pad_len, pad_len]], constant_values=0)
    center[0] = (center[0] - xmi) + pad_len
    center[1] = (center[1] - ymi) + pad_len

    for angle in range(-45, 45):
        vals[angle] = _rotate_and_score(msk_padded.copy(), angle, center, score_func)

    return min(vals, key=vals.get)
