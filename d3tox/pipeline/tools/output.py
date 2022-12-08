from d3tox.utils.nii import save_nii_take_headers_from


def save_img_and_msk(img_msk, take_headers_from_nii_path, output_dir, name):
    for k, v in img_msk.items():
        img = v[0]
        msk = v[1]

        # assure everything > 0 is set to 1 for the mask
        msk[msk > 0] = 1

        save_nii_take_headers_from(img, take_headers_from_nii_path, output_dir / f'{name}-{k}-img.nii.gz')
        save_nii_take_headers_from(msk, take_headers_from_nii_path, output_dir / f'{name}-{k}-msk.nii.gz')
