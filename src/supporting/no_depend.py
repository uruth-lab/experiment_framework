# Created to hold misc functions that do not depend on any other project files
import re
import time
from typing import Any, Type


def get_valid_ident(org):
    """Turns the parameter passed into a valid identifier"""
    result = re.sub(r' ', '_', str(org))
    result = re.sub(r'[^A-Za-z0-9_]', '', result)
    return result


def public_members_as_dict(class_: Type[Any]):
    result = {}
    for i in class_.__dict__.items():
        if not i[0][0] == "_":
            result[i[0]] = i[1]
    return result


def get_run_timestamp():
    return time.strftime("%Y-%m-%d %H%M%S")
