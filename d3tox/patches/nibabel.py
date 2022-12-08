# monkeypatches for nibabel
import inspect
import nibabel.loadsave


def _signature_matches_extension(filename, *args, **kwargs):
    """Check if signature aka magic number matches filename extension.
    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file to check
    sniff : bytes or None
        First bytes of the file. If not `None` and long enough to contain the
        signature, avoids having to read the start of the file.
    Returns
    -------
    matches : bool
       - `True` if the filename extension is not recognized (not .gz nor .bz2)
       - `True` if the magic number was successfully read and corresponds to
         the format indicated by the extension.
       - `False` otherwise.
    error_message : str
       An error message if opening the file failed or a mismatch is detected;
       the empty string otherwise.
    """

    return _org_signature_matches_extension(filename, None)


# Older versions have a sniff variable which is improperly set. We want to ignore it because it throws exceptions.
# This bug occurs when a .gz / .bz2 / .zst without a nii is handed to the function.
if tuple([int(x) for x in nibabel.__version__.split('+')[0].split('.')]) < (5, 0, 0):
    if 'sniff' in inspect.signature(nibabel.loadsave._signature_matches_extension).parameters:
        _org_signature_matches_extension = nibabel.loadsave._signature_matches_extension
        nibabel.loadsave._signature_matches_extension = _signature_matches_extension
