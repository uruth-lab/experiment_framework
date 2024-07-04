import ntpath
import os
from pathlib import Path


def mkdir_w_par(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)


def change_filename_ext(fn: str, new_ext: str) -> str:
    """
    Returns the filename with the next extension
    :param fn: The filename with the old extension
    :param new_ext: The new extension to put on the filename
    :return: The filename with the new extension
    """
    result, _ = os.path.splitext(fn)
    return f'{result}{new_ext}'


def ensure_folder_created(fn: str):
    """
    Expects fn to be a file name. Removes the file name and ensures that
    it's parent folder is exists.
    :param fn: Filename to use to find parent folder
    :return:
    """
    folder, _ = ntpath.split(fn)
    mkdir_w_par(folder)
