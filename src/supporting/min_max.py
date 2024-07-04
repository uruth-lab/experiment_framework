from typing import Optional, List

import numpy as np


class MinMax:
    def __init__(self, min_: Optional[float] = None, max_: Optional[float] = None):
        self.min: float = np.inf if min_ is None else min_
        self.max: float = -np.inf if max_ is None else max_

    def __repr__(self):
        return f"[{self.min},{self.max}]"


def get_min_max_from_data(data, leading_features_included: Optional[int] = None) -> List[MinMax]:
    """
    Gets the min and max from `data` for `leading_features_included`. If `leading_features_included` is not specified
    all features are used
    :param data: The data to do the extraction from
    :param leading_features_included: The number of features from the start to use (or all if not provided)
    :return:
    """
    if leading_features_included is None:
        leading_features_included = 1 if len(data.shape) == 1 else data.shape[1]
    result = [MinMax() for _ in range(leading_features_included)]
    for x in data:
        for index_feature, value in enumerate(x[:leading_features_included]):
            if result[index_feature].min > value:
                result[index_feature].min = value
            if result[index_feature].max < value:
                result[index_feature].max = value
    return result
