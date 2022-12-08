import numpy as np


def crop_from_mask(img, msk, border=.1):
    maxcoords = np.amax(np.where(msk > 0), axis=1).astype(int)
    mincoords = np.amin(np.where(msk > 0), axis=1).astype(int)

    mincoords = np.round((mincoords * (1 - border))).astype(int)
    maxcoords = np.round((maxcoords * (1 + border))).astype(int)


    #crop_coords = (mincoords[0], maxcoords[0] + 1, mincoords[1], maxcoords[1] + 1)
    #crop_coords = np.empty((mincoords.size + maxcoords.size,), dtype=mincoords.dtype)
    #crop_coords[0::2] = mincoords
    #crop_coords[1::2] = maxcoords

    # create tuple of slices for advanced ndim-independent numpy slicing
    crop_coords = tuple([slice(mincoords[i], maxcoords[i] + 1) for i in range(len(mincoords))])


    # if img.ndim == 2:
    #     img_cropped = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
    #     msk_cropped = msk[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
    # elif img.ndim == 3:
    #     img_cropped = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], crop_coords[4]:crop_coords[5]]
    #     msk_cropped = msk[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], crop_coords[4]:crop_coords[5]]

    img_cropped = img[crop_coords]
    msk_cropped = msk[crop_coords]

    return img_cropped, msk_cropped
