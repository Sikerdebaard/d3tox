import numpy as np
import copy

from pathlib import Path
from PIL import Image, ImageDraw
from skimage.morphology import erosion

import d3tox.pipeline.tools.transform as transform
from d3tox.pipeline.tools.vertebrae import vertebrae_colors


def write_png_img(img, file, is_rgb=False):
    file = Path(file)

    # if 3d image call imsave for each slice
    if img.ndim == 3 and not is_rgb:
        for i in range(img.shape[2]):
            fname = file.parent / f'{file.stem}_{i:05d}{".".join(file.suffixes)}'
            write_png_img(img[:, :, i], fname)

        return

    if not is_rgb:
        img = to_rgb(img)

    im = Image.fromarray(img)
    im.save(str(file))


def to_rgb(img, alpha=False):
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img

    # scale between 0 - 255
    img_rgb *= (255.0 / img_rgb.max())

    if alpha:
        # add alpha layer
        img_rgb = np.dstack(( img_rgb, np.zeros((img_rgb.shape[0:2]))))
        img_rgb[:, :, 3] = 255

    return img_rgb.astype(np.uint8)


def to_pil_image(img):
    if img.ndim != 3:
        img = to_rgb(img)

    return Image.fromarray(img)


def draw(img):
    dimg = ImageDraw.Draw(img, 'RGB')

    return dimg


def draw_mask(dimg, msk, color, edge_only=True):
    if edge_only:
        eroded = erosion(msk)
        msk[eroded > 0] = 0

    msk[msk > 0] = 255
    msk = msk.astype(np.uint8)

    pil_msk = Image.fromarray(msk, mode='L')

    dimg.bitmap((0, 0), pil_msk, fill=color)

    return dimg


def draw_centroids(dimg, centroids_xy, colors, r=3):
    for i in range(len(centroids_xy)):
        color = colors[i]
        x, y = centroids_xy[i]
        dimg.ellipse((x-r, y-r, x+r, y+r), fill=color)

    return dimg


def draw_line_on_image(dimg, xy, color=(255, 0, 0)):
    xy = xy[np.argsort(xy[:, 0])]
    c = [tuple([c[0], c[1]]) for c in xy]
    dimg.line(c, fill=color, width=1)

    return dimg


def draw_and_save_intermediate1(xray_img, segmentations, spline_xy, centroids_xy, subject, outdir, image_format='png'):
    xray_img = xray_img.copy()
    segmentations = copy.deepcopy(segmentations)

    pimg = to_pil_image(xray_img)
    dimg = draw(pimg)

    dimg = draw_line_on_image(dimg, spline_xy)

    for segname, msk in segmentations.items():
        dimg = draw_mask(dimg, msk, color=vertebrae_colors[segname], edge_only=True)

    draw_centroids(dimg, centroids_xy, colors=[vertebrae_colors[k] for k in sorted(vertebrae_colors.keys())])

    pimg = np.array(pimg)

    pimg = _crop_to_masks(pimg, list(segmentations.values()))

    # fname = subj_dir / 'spline.png'
    fname = outdir / f'1_{subject}.{image_format}'
    write_png_img(pimg, fname, is_rgb=True)


def _crop_to_masks(img, masks):
    msk = np.zeros(masks[0].shape)
    for mask in masks:
        msk[np.where(mask > 0)] = 255

    slicer_y, slicer_x, new_center = transform.calc_crop_slicer_from_mask(msk, [0, 0], perc_extra=.02)

    return img[slicer_y, slicer_x]


def draw_and_save_intermediate2(cropped_and_rotated_img_msk, subject, outdir, image_format='png'):
    cropped_and_rotated_img_msk = copy.deepcopy(cropped_and_rotated_img_msk)

    for k, images in cropped_and_rotated_img_msk.items():
        img, msk, cent, negangle, opti_angles = images

        pimg = to_pil_image(img)
        dimg = draw(pimg)

        if k in vertebrae_colors:
            color = vertebrae_colors[k]
        else:
            color = (0, 255, 0)

        draw_mask(dimg, msk, color=color, edge_only=True)
        draw_centroids(dimg, [cent], colors=[color])

        pimg = np.array(pimg)

        fname = outdir / f'2_{subject}_{k}.{image_format}'
        write_png_img(pimg, fname, is_rgb=True)


def draw_and_save_intermediate3(xray_img, segmentations, centers_xy, negated_corners, subject, outdir, image_format='png'):
    xray_img = xray_img.copy()
    segmentations = copy.deepcopy(segmentations)

    pimg = to_pil_image(xray_img)
    dimg = draw(pimg)

    for segname, msk in segmentations.items():
        dimg = draw_mask(dimg, msk, color=vertebrae_colors[segname], edge_only=True)

    draw_centroids(dimg, centers_xy, colors=[vertebrae_colors[k] for k in sorted(vertebrae_colors.keys())])

    corners = []
    colors = []
    for name, vbra in negated_corners.items():
        col = tuple([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
        for cname, corner in vbra.items():
            corners.append([corner[0], corner[1]])
            colors.append(col)

    draw_centroids(dimg, corners, colors=colors)

    pimg = np.array(pimg)

    pimg = _crop_to_masks(pimg, list(segmentations.values()))

    # fname = subj_dir / 'spline.png'
    fname = outdir / f'3_{subject}.{image_format}'
    write_png_img(pimg, fname, is_rgb=True)


def DEBUG_draw_masks(masks, colors, fname, edge_only=False, use_counter=True):
    return

    fname = Path(fname)
    if use_counter:
        global debug_draw_mask_counter
        fname = fname.parent / f'{fname.stem}_{debug_draw_mask_counter}{".".join(fname.suffixes)}'
        debug_draw_mask_counter += 1

    pimg = Image.new("RGB", masks[0].shape[::-1], (0, 0, 0))
    dimg = draw(pimg)

    for i in range(len(masks)):
        msk = masks[i]
        color = colors[i]
        dimg = draw_mask(dimg, msk, color=color, edge_only=edge_only)

    pimg = np.array(pimg)
    write_png_img(pimg, fname, is_rgb=True)


debug_draw_mask_counter = 0
