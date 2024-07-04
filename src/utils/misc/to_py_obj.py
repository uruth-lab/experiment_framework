from typing import List


def getattr_multi_level(obj, name: str):
    """
    Iterates through the name to get the target attribute starting at
    obj. Can go down multiple levels to get the attribute
    :param obj: The base object to get name from
    :param name: The name of desired attribute (can be multiple levels deep)
    :return: The desired attribute
    """
    names = name.split('.')
    for x in names:
        obj = getattr(obj, x)
    return obj


def str_to_class(cls_name: str) -> type:
    """
    Converts a string into the class that it represents

    NB: Code based on https://stackoverflow.com/questions/452969/does-python
    -have-an-equivalent-to-java-class-forname
    :param cls_name: The string representation of the desired class
    :return: A pointer to the class (Able to be used as a constructor)
    """
    parts = cls_name.split('.')
    modules = '.'.join(parts[:-1])
    result = __import__(modules)
    for comp in parts[1:]:
        result = getattr(result, comp)
    return result


def strs_to_classes(cls_names: List[str]) -> List[type]:
    result = []
    for s in cls_names:
        result.append(str_to_class(s))
    return result
