from d3tox.utils.nii import load_nii, get_nii_mask_largest_blob, get_nii_mask_center_of_mass

import pandas as pd

def find_mask_center_of_mass(msk):
    centroid = get_nii_mask_center_of_mass(msk)[0]
    return centroid


def find_nii_mask_center_of_mass(nii_files):
    results = {}
    for name, nii_file in nii_files.items():
        nii = load_nii(nii_file)

        msk = get_nii_mask_largest_blob(nii)
        centroid = get_nii_mask_center_of_mass(msk)[0]

        results[name] = centroid

    return pd.DataFrame(results)
