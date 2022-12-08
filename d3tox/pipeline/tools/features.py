from d3tox.pipeline.tools.centers import find_mask_center_of_mass

import numpy as np

from d3tox.pipeline.tools.vertebrae import _euclidean_dist


def disc_features(discs, c2_msk):
    # first get the height of C2 as this vertebrae almost never has degeneration
    c2_height = _vbra_width_height(c2_msk, _msk_mpoint(c2_msk))[1]

    named_feats = {}
    for k, msk in discs.items():
        feats = _disc_lenght_based_features(msk)

        # normalize length features to c2 height (calc ratio)
        # and put in named_feats
        named_feats[k] = {fname: v / c2_height for fname, v in feats.items()}

    return named_feats


def _disc_lenght_based_features(msk):
    mpoint = _msk_mpoint(msk)

    width, height = _vbra_width_height(msk, mpoint)

    return {'disc_width': width, 'disc_height': height}


def vertebrae_features(vertebraes, vertebrae_corners_yx):
    # first get the height of C2 as this vertebrae almost never has degeneration
    c2_msk = vertebraes['C2']
    c2_height = _vbra_width_height(c2_msk, _msk_mpoint(c2_msk))[1]

    named_feats = {}
    for k, msk in vertebraes.items():
        if k == 'C2':
            continue  # ignore C2

        # calculate length features
        feats = _vbra_length_based_features(msk, vertebrae_corners_yx[k])

        # normalize length features to c2 height (calc ratio)
        # and put in named_feats
        named_feats[k] = {fname: v / c2_height for fname, v in feats.items()}

    return named_feats


def _vbra_length_based_features(msk_vertebrae, corners_yx):
    mpoint = _msk_mpoint(msk_vertebrae)

    width, height = _vbra_width_height(msk_vertebrae, mpoint)

    named_feats = {'vbra_width': width, 'vbra_height': height}

    for k in sorted(corners_yx.keys()):
        apy = (mpoint[0], corners_yx[k][0])
        apx = (mpoint[1], corners_yx[k][1])
        named_feats[f'{k}_midpoint_dist_y'] = max(apy) - min((apy))
        named_feats[f'{k}_midpoint_dist_x'] = max(apx) - min((apx))

        euclid = _euclidean_dist(np.array([corners_yx[k]]), np.array([mpoint]))
        assert len(euclid) == 1
        named_feats[f'{k}_midpoint_dist_euclid'] = euclid[0]

    return named_feats


def _msk_mpoint(msk):
    mpoint = find_mask_center_of_mass(msk)
    mpoint = np.round(mpoint, 0).astype(int)

    return mpoint


def _vbra_width_height(msk, mpoint):
    tmp = np.where(msk[:, mpoint[1]] > 0)
    height = (np.max(tmp) - np.min(tmp)) + 1

    tmp = np.where(msk[mpoint[0], :] > 0)
    width = (np.max(tmp) - np.min(tmp)) + 1

    return width, height

