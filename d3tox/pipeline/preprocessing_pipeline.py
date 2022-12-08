import numpy as np
from joblib import Parallel, delayed
from nibabel.filebasedimages import ImageFileError

from tqdm.auto import tqdm

from d3tox.pipeline.tools.features import vertebrae_features, disc_features
from d3tox.pipeline.tools.filesearch import search_valid_datafiles
from d3tox.pipeline.tools.centers import find_nii_mask_center_of_mass
from d3tox.pipeline.tools.orientation import estimate_img_orientation

import logging
import pandas as pd

from d3tox.pipeline.tools.output import save_img_and_msk
from d3tox.pipeline.tools.rotation import rotate_2d_points_over_center
from d3tox.pipeline.tools.segment import segment_discs
from d3tox.pipeline.tools.tabular import named_feats_to_csv
from d3tox.pipeline.tools.transform import rotate, rotate_and_crop_to_msk
from d3tox.pipeline.tools.vertebrae import estimate_vertebrae_angles, find_vertebrae_corners, negate_corner_rotations
from d3tox.utils.draw import draw_and_save_intermediate1, draw_and_save_intermediate2, draw_and_save_intermediate3
from d3tox.utils.joblib import tqdm_joblib
from d3tox.utils.nii import get_nii_file_ndim_shape, load_nii, get_nii_img, count_nii_mask_blobs


def _prefilight_checks(df_files, indir):
    # pre-flight checks
    error = False

    min_subjects_needed = 2
    if df_files.shape[0] < min_subjects_needed:
        logging.error(f'A minimum of {min_subjects_needed} subjects are needed, only {df_files.shape[0]} subjects were found')
        error = True

    for idx, row in df_files.iterrows():
        for col in df_files.columns:
            file = row[col]

            if pd.isna(file):
                continue

            ndim, shape = get_nii_file_ndim_shape(indir / file)
            if (ndim == 3 and shape[2] != 1) or (ndim > 3) or (ndim <= 1):
                if col != 'XRAY':
                    logging.error(f'This tool only support 2d images, the following file is a {ndim}d volume: {indir / file}')
                    error = True
                else:
                    # Probably an XRAY stored as RGB
                    if shape[-1] != 3 and shape[-1] != 1:
                        # image is not likely rgb or grayscale so we throw an error to the user
                        logging.error(f'XRAY image is an {ndim}d volume shaped as {shape}, this is unsupported: {indir / file}')
                        error = True

    return error


def _exclude_invalid_files(df_files, indir):
    exclude = {}
    nan = float('nan')
    exclude_messages = []
    logging.info('Checking files...')
    for idx, row in tqdm(tuple(df_files.iterrows()), unit='Subjects'):
        for col in df_files.columns:
            file = row[col]
            if pd.isna(file):
                continue

            try:
                f = load_nii(indir / file)
            except ImageFileError as e:
                logging.error(str(e))
                exclude[file] = nan
                continue

            if col == 'XRAY':
                pass
            else:
                blobs = count_nii_mask_blobs(f)
                if blobs > 1:
                    exclude_messages.append(f'Excluding file found more than 1 blob and/or mask in file {indir / file}')
                    exclude[file] = nan
                elif blobs == 0:
                    exclude_messages.append(f'Excluding file no mask found in {indir / file}')
                    exclude[file] = nan

    for msg in exclude_messages:
        logging.info(msg)

    return df_files.replace(exclude)


def _match_inclusion_criteria(df_files, vertebrae_cols, xray_col):
    # exclude all subjects without XRAY imaging
    excluded = set(df_files.index[df_files[xray_col].isna()])
    if len(excluded) > 0:
        logging.warning(
            f'The following subjects were excluded due to a missing XRAY image: {", ".join(excluded)}')

    # drop all subjects with < 3 vertebrae segmentations
    sel = df_files[vertebrae_cols].dropna(thresh=3, axis=0).index
    excl_sel = df_files.index[~df_files.index.isin(sel)]
    excluded = excluded | set(excl_sel)
    if len(excl_sel) > 0:
        logging.warning(
            f'The following subjects were excluded because < 3 vertebrae segmentations were found: {", ".join(excl_sel)}'
        )

    excl_sel = df_files.index[df_files['C2'].isna()]

    excluded = excluded | set(excl_sel)
    if len(excl_sel) > 0:
        logging.warning(
            f'The following subjects were excluded because the C2 vertebrae segmentation was not found: {", ".join(excl_sel)}'
        )

    excl_sel = df_files[df_files[df_files.columns.drop(['XRAY', 'C7'])].isna().any(axis=1)].index
    excluded = excluded | set(excl_sel)
    if len(excl_sel) > 0:
        logging.warning(
            f'The following subjects were excluded because the vertebrae are non-continues (C3 or C4 or C5 or C6 are missing): {", ".join(excl_sel)}'
        )

    df_files = df_files.loc[~df_files.index.isin(excluded)]

    return df_files


def _process_subject(indir, outdir, subject, df_files, vertebrae_cols, x_idx, y_idx, intermediates_dir, extra_optimizations, debug):
    subj_dir = outdir / 'subjects' / subject
    subj_dir.mkdir(exist_ok=True, parents=True)

    files = {k: indir / v for k, v in df_files.loc[subject, vertebrae_cols].dropna().to_dict().items()}
    xray_file = df_files.loc[subject, 'XRAY']

    # find the centroids of the segmentation masks
    centers = find_nii_mask_center_of_mass(files)

    # estimate the image orientation, rotate centroids
    # note: midpoint_xy is in x, y notation and does not follow x_idx / y_idx
    rot_radians, midpoint_xy = estimate_img_orientation(centers, x_idx, y_idx)

    # note: returned centroids_xy var is in x, y notation and does not follow x_idx / y_idx
    centers_xy = rotate_2d_points_over_center(centers.loc[[x_idx, y_idx]].T.values, midpoint_xy, rot_radians)

    vertebrae_radians, spline_xy = estimate_vertebrae_angles(centers_xy)

    xray_img = get_nii_img(load_nii(indir / xray_file))
    xray_img = rotate(xray_img, np.degrees(rot_radians), midpoint_xy, 3)

    segmentations = {
        k: rotate(get_nii_img(load_nii(v)), np.degrees(rot_radians), midpoint_xy.copy(), 0) for k, v in files.items()
    }

    draw_and_save_intermediate1(xray_img, segmentations, spline_xy, centers_xy, subject, intermediates_dir)

    # TODO: this sometimes crashes if a vertebrae is missing
    cropped_and_rotated_img_msk = {
        k: rotate_and_crop_to_msk(
            xray_img.copy(), msk.copy(), vertebrae_radians[vertebrae_cols.tolist().index(k)],
            centers_xy[vertebrae_cols.tolist().index(k)].copy(), k,
            extra_optimizations)
        for k, msk in segmentations.items()
    }

    draw_and_save_intermediate2(cropped_and_rotated_img_msk, subject, intermediates_dir)

    corners_yx = {k: find_vertebrae_corners(v[1], v[2]) for k, v in cropped_and_rotated_img_msk.items()}

    negation_angles = {k: v[3] for k, v in cropped_and_rotated_img_msk.items()}
    vbra_angles_optimized = {k: v[4] for k, v in cropped_and_rotated_img_msk.items()}
    new_vertebrae_centers_xy = {k: v[2] for k, v in cropped_and_rotated_img_msk.items()}
    old_vertebrae_centers_xy = {k: centers_xy[vertebrae_cols.tolist().index(k)].copy() for k in segmentations.keys()}
    negated_corners = {k: negate_corner_rotations(corner, negation_angles[k], new_vertebrae_centers_xy[k], old_vertebrae_centers_xy[k]) for k, corner in corners_yx.items()}

    draw_and_save_intermediate3(xray_img, segmentations, centers_xy, negated_corners, subject, intermediates_dir)

    msk_discs, angles_disc_guess_rads, centers_discs_yx = segment_discs(negated_corners, segmentations, vbra_angles_optimized)

    cropped_and_rotated_disc_img_msk = {
        k: rotate_and_crop_to_msk(
            xray_img.copy(), msk.copy(), angles_disc_guess_rads[k],
            # transform yx into xy
            centers_discs_yx[k][::-1], k,
            extra_opti=False)
        for k, msk in msk_discs.items()
    }

    draw_and_save_intermediate2(cropped_and_rotated_disc_img_msk, subject, intermediates_dir)

    vbra_feats = vertebrae_features({k: v[1] for k, v in cropped_and_rotated_img_msk.items()}, corners_yx)
    disc_feats = disc_features({k: v[1] for k, v in cropped_and_rotated_disc_img_msk.items()}, cropped_and_rotated_img_msk['C2'][1])

    named_feats_to_csv(vbra_feats, subj_dir / 'vbra-feats.csv')
    named_feats_to_csv(disc_feats, subj_dir / 'disc-feats.csv')

    save_img_and_msk(cropped_and_rotated_img_msk, indir / xray_file, subj_dir, 'vbra')
    save_img_and_msk(cropped_and_rotated_disc_img_msk, indir / xray_file, subj_dir, 'disc')


def run_pipeline(indir, outdir, case_insensitive, xray, c2, c3, c4, c5, c6, c7, extra_optimizations, workers=1, DEBUG=False):
    df_files, errors = search_valid_datafiles(indir, outdir / 'find_files', case_insensitive, xray, c2, c3, c4, c5, c6, c7)

    df_files = _exclude_invalid_files(df_files, indir)

    vertebrae_cols = df_files.columns[1:]
    xray_col = df_files.columns[0]

    df_files = _match_inclusion_criteria(df_files, vertebrae_cols, xray_col)

    print('Running preflight checks...')
    if _prefilight_checks(df_files, indir):
        return 1  # if the preflight checks fail we stop running

    print('Preflight checks completed!')

    n_subject = df_files.shape[0]
    n_images = df_files[xray_col].shape[0]
    n_masks = {x: df_files[x].dropna().shape[0] for x in vertebrae_cols}

    masks_msg = ', '.join([f'{v} {k} segmentations' for k, v in n_masks.items()])

    logging.info(f'The following subjects will be included: {", ".join(df_files.index)}')
    logging.info(f'Found {n_subject} subjects, {n_images} images, {masks_msg}')

    x_idx = 1
    y_idx = 0

    intermediates_dir = outdir / 'intermediates'
    intermediates_dir.mkdir(exist_ok=True, parents=True)

    # run the pipeline
    with tqdm_joblib(tqdm(total=df_files.shape[0], unit=' subjects', leave=True)) as progress_bar:
        retvals = Parallel(n_jobs=workers)(
            delayed(_process_subject)(indir, outdir, subject, df_files, vertebrae_cols, x_idx, y_idx, intermediates_dir, extra_optimizations, debug=DEBUG) for subject in df_files.index
        )

