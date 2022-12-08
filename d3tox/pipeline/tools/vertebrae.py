import scipy.interpolate as interpolate
import numpy as np
from skimage.morphology import erosion
from skimage.transform import resize
import d3tox
from d3tox.pipeline.tools.rotation import rotate_2d_points_over_center

vertebrae_colors = {
    'C2': (21, 127, 31),
    'C3': (76, 44, 105),
    'C4': (160, 234, 222),
    'C5': (254, 198, 1),
    'C6': (251, 172, 190),
    'C7': (149, 25, 12),
}


def estimate_vertebrae_angles(center_xy):
    # sort over x-axis
    sorted_indices = np.argsort(center_xy[:, 0])
    xy_sorted = center_xy[sorted_indices]

    x = xy_sorted[:, 0]
    y = xy_sorted[:, 1]

    # t, c, k = interpolate.splrep(x=x, y=y, s=0, k=5 if xy_sorted.shape[0] > 5 else xy_sorted.shape[0] - 1)
    t, c, k = interpolate.splrep(x=x, y=y, s=0, k=1)

    xmi, xma = x[0], x[-1]
    N = np.round(xma - xmi).astype(int)
    x_interp = np.linspace(xmi, xma, N)
    y_interp = interpolate.splev(x_interp, (t, c, k), der=0)

    der = interpolate.splev(x, (t, c, k), der=1)
    radians = np.arctan(der)

    # change indices back to centroids_xy so that it alligns properly (C2 index 0, C3 index 1 etc.)
    radians = radians[np.argsort(sorted_indices)]

    return radians, np.column_stack([x_interp, y_interp])


def _euclidean_dist(origin_yx, coords):
    dist = coords - origin_yx
    dist = np.sqrt(np.sum(dist ** 2, axis=1))

    return dist


def find_vertebrae_corners(msk, center_xy):
    edge = msk.copy()
    edge[erosion(edge) > 0] = 0
    edge[edge > 0] = 255

    center_xy = np.round(center_xy).astype(int)

    quarts = dict(
        top_left=edge[0:center_xy[1], 0:center_xy[0]],
        top_right=edge[0:center_xy[1], center_xy[0]:edge.shape[1]],
        bottom_left=edge[center_xy[1]:edge.shape[0], 0:center_xy[0]],
        bottom_right=edge[center_xy[1]:edge.shape[0], center_xy[0]:edge.shape[1]],
    )

    origins_xy = dict(
        top_left=[quarts['top_left'].shape[1]-1, quarts['top_left'].shape[0]-1],
        top_right=[0, quarts['top_right'].shape[0]-1],
        bottom_left=[quarts['bottom_left'].shape[1]-1, 0],
        bottom_right=[0, 0],
    )

    coords_converter_xy = dict(
        top_left=[0, 0],
        top_right=[center_xy[0], 0],
        bottom_left=[0, center_xy[1]],
        bottom_right=[center_xy[0], center_xy[1]]
    )

    corners = {}
    for name, qmsk in quarts.items():
        idx_mask_mi, idx_mask_ma = np.argmin(qmsk.shape), np.argmax(qmsk.shape)
        mask_mi, mask_ma = qmsk.shape[idx_mask_mi], qmsk.shape[idx_mask_ma]
        msk_res = resize(qmsk, (mask_ma, mask_ma), order=0)
        msk_coords = np.where(msk_res > 0)
        msk_coords = np.column_stack([msk_coords[0], msk_coords[1]])

        origins_yx = [origins_xy[name][1], origins_xy[name][0]]
        origins_yx[idx_mask_mi] = origins_yx[idx_mask_mi] / mask_mi * mask_ma

        dist = _euclidean_dist(origins_yx, msk_coords)

        # take the top=10 of corners and calculate the mean
        # idx_min = (-dist).argsort()[:10]
        # corner = msk_coords[idx_min].mean(axis=0).round(0).astype(int)
        corner = msk_coords[(-dist).argsort()[0]]

        corner[idx_mask_mi] = np.round(corner[idx_mask_mi] / mask_ma * mask_mi, 0).astype(int)

        corner = corner[0] + coords_converter_xy[name][1], corner[1] + coords_converter_xy[name][0]

        corners[name] = corner


    # DEBUG: DRAWER
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    msk_corners = []
    for name, qmsk in quarts.items():
        mask_origin = np.zeros(qmsk.shape, dtype=np.uint8)
        mask_origin[origins_xy[name][1], origins_xy[name][0]] = 255

        mask_corner = np.zeros(qmsk.shape, dtype=np.uint8)
        corner = corners[name]
        corner = corner[0] - coords_converter_xy[name][1], corner[1] - coords_converter_xy[name][0]
        s = 2
        mask_corner[corner[0] - s:corner[0] + s, corner[1] - s:corner[1] + s] = 255

        colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255)]

        d3tox.utils.draw.DEBUG_draw_masks([qmsk.copy(), mask_corner, mask_origin], colors, f'/home/thomas/tmp/cervout/test/vert_{name}.png')

        # second debug drawing

        mask_corner = np.zeros(msk.shape, dtype=np.uint8)
        corner = corners[name]
        s = 2
        mask_corner[corner[0] - s:corner[0] + s, corner[1] - s:corner[1] + s] = 255
        msk_corners.append(mask_corner)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    d3tox.utils.draw.DEBUG_draw_masks([msk, *msk_corners], colors, f'/home/thomas/tmp/cervout/test/QQ_{name}.png')

    # DEBUG: DRAWER < />

    return corners


def negate_corner_rotations(corners, negation_angle, centerpoint_xy, original_centerpoint_xy):
    corners_out = {}

    center_xy_diff = np.array(original_centerpoint_xy) - np.array(centerpoint_xy)

    for name, corner in corners.items():
        res = rotate_2d_points_over_center([[corner[1], corner[0]]], np.array(centerpoint_xy), np.radians(negation_angle))
        res = res[0]
        res[0] = res[0] + center_xy_diff[0]
        res[1] = res[1] + center_xy_diff[1]
        corners_out[name] = res

    return corners_out
