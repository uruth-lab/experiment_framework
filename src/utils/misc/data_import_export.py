import csv
import logging
import re
from typing import Any, List, Type

import yaml

from src.utils.misc.files_folders import ensure_folder_created

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def get_matlab_safe_list(org_list: list):
    """
    Take a list and returns a new list that is matlab safe.

    :param org_list: input list to be made safe
    :return: list that is afe to save in matlab
    """
    result = []
    for val in org_list:
        if val is None:
            val = str(val)
        elif isinstance(val, dict):
            val = get_matlab_safe_dict(val)
        result.append(val)
    return result


def get_matlab_safe_dict(org_dict: dict):
    """
    Takes a dict and returns a new dict that is matlab safe.
    Removes None and validates keys.

    NB: keys may collide if the resolve to the same "clean" string. Warning
    logged but conversion not aborted.

    :param org_dict: input dict to be made safe
    :return: dict that is safe to save in matlab
    """
    result = {}
    for key in org_dict.keys():
        valid_key = get_valid_ident(key)
        value = org_dict[key]
        if value is None:
            value = str(value)
        elif isinstance(value, dict):
            value = get_matlab_safe_dict(value)
        elif isinstance(value, list):
            value = get_matlab_safe_list(value)

        if result.get(valid_key) is not None:
            logging.warning(
                f'Got DUPLICATE key in dict with str rep of "{valid_key}"')
        result[valid_key] = value

    return result


def get_valid_ident(org):
    """Turns the parameter passed into a valid identifier"""
    result = re.sub(r' ', '_', str(org))
    result = re.sub(r'[^A-Za-z0-9_]', '', result)
    return result


def public_members_as_dict(class_: Type[Any]) -> dict:
    """
    Converts a class into a dict of its public values
    :param class_: the class to be converted
    :return: the public values from the class
    """
    result = {}
    for i in class_.__dict__.items():
        if not (i[0][0] == "_" or isinstance(i[1], classmethod)):
            if not isinstance(i[1], type):
                result[i[0]] = i[1]
            else:
                result[i[0]] = public_members_as_dict(i[1])
    return result


def save_list_to_file(file_list: List[str], file_name: str, end: str = ''):
    """
    Saves file_list to a file with file_name and appends end
    :param file_list: List of string to be saved to the file
    :param file_name: Name of file to be created or overwritten
    :param end: Optional line ending to append to each line
    :return:
    """
    ensure_folder_created(file_name)
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write('%s%s' % (item, end))


def save_yaml(data, fn: str):
    """
    Saves data to yaml file named fn. Overwrites if it already exists
    :param data: Data to be saved
    :param fn: The name of the file to create
    """
    ensure_folder_created(fn)
    with open(fn, 'w') as f:
        f.write(yaml.dump(data, Dumper=Dumper))


def load_yaml(fn: str):
    """
    Loads data from a yaml file named fn
    :param fn: The name of the file to load from
    :return: The data from the file
    """
    with open(fn, 'r') as f:
        return yaml.load(f, Loader=Loader)


def _latex_sanitize(s: str) -> str:
    """
    Fixes common problems when trying to generate latex
    :param s: The input text to sanitize
    :return: The sanitized text
    """
    return s.replace("_", "\\_")  # Replace bare underscores for latex


def _latex_wrap_table_body(body: str, data: List[List]) -> str:
    col_widths = latex_estimate_column_widths(data)
    prefix = "\\begin{center}\n"
    prefix += "    \\begin{tabular}{ |" + col_widths + " }\n        \\hline\n"
    suffix = """    \\end{tabular}
\\end{center}\n"""
    return _latex_sanitize(prefix + body + suffix)


def latex_estimate_column_widths(data: List[List]) -> str:
    # Assumes first row is made up of strings (makes code simpler as we need starting value for each col)
    max_width_per_col = [len(cell) for cell in data[0]]
    for row in data[1:]:
        for i, cell in enumerate(row):
            if isinstance(cell, str):
                if len(cell) > max_width_per_col[i]:
                    max_width_per_col[i] = len(cell)
    final_widths = [max(2.0, x / 2) for x in max_width_per_col]
    result = ""
    for width in final_widths:
        result += f" m{{{width}em}} |"
    return result


def latex_convert_to_table(data: List[List]) -> str:
    result = ""
    for row_idx, row in enumerate(data):
        line = ""
        for cell_idx, cell in enumerate(row):
            # Header row done here because there is too much machinery to duplicate if moving it
            if row_idx == 0:
                line += "\\textbf{"
            line += f"{cell}" if not isinstance(cell, float) else f"{cell:.3f}"
            if row_idx == 0:
                line += "}"
            if cell_idx < len(row) - 1:
                line += " & "
        result += f"        {line} \\\\ \\hline\n"
    result = _latex_wrap_table_body(result, data)
    return result


def csv_write_rows(data: List[List], fn: str):
    with open(fn, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator="\n")
        for row in data:
            csv_writer.writerow(row)
