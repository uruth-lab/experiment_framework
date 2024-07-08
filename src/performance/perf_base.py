import math
from abc import abstractmethod
from collections.abc import Iterable
from numbers import Number
from typing import List, Union, Optional, Tuple, Collection

import numpy as np
from sklearn import metrics

from src.config import Conf
from src.supporting.no_depend import get_valid_ident
from src.utils.misc.files_folders import ensure_folder_created


class PerfConfig:
    # TODO : 1 Move to Conf
    # Cannot be in main config because of cyclic import
    VIZ_DISPLAY = not True
    VIZ_USE_FOLDER_PER_EXP = not True
    VIZ_SAVE_TO_DISK = True
    UNDEFINED_VALUE = -1  # Used when a metric is mathematically undefined
    VIZ_LINE_WEIGHT = 2
    VIZ_DPI = 300
    VALUE_FAIL = -88  # Used when an approach fails to find a value
    APX_EQUAL_THRESHOLD = 1e-7
    OUTSIDE_OF_DATA_BAND_CONTINUOUS_CHECK_VALUES = (2, 10, 100)
    RELEASE_MODE = not True
    SHOW_HEADINGS = not True and not RELEASE_MODE
    LONG_LABELS = True

    class ScatterPlot:
        SAME_SCALE = True
        EQUAL_ASPECT = True
        COLOR_TP = '#0000FF'
        COLOR_FP = '#FFA500'
        COLOR_TN = '#8888FF'
        COLOR_FN = '#FF7700'
        COLOR_EXTRA = '#0000FF'
        HEATMAP_COLORMAP = 'YlGn'
        MARKER_TP = '*'
        MARKER_FP = '+'
        MARKER_TN = '+'
        MARKER_FN = '*'
        MARKER_RAW = 'x'
        MARKER_EXTRA = 'o'
        HEATMAP_ENABLED = True
        HEATMAP_POINTS_PER_AXIS = 350
        OPEN_SPACE_PERCENTAGE = 50
        COLOR_BAR_TICK_COUNT = 5
        FLOOR_SCORES_FOR_COLORS_ONLY_TO_TICKS = not True

        ARTIFICIALLY_BOOST_ABOVE_THRESHOLD = not True
        "Used to make heatmap more clearly separated (Only has an effect when floor is on)"
        INCLUDE_HEATMAP_BOUNDARY_VALUES_AS_TICKS = True
        "Controls if points like the min/max of the heatmap get added as ticks"
        INCLUDE_HEATMAP_THRESHOLD_AS_TICK = True
        "Controls if the threshold value gets added as ticks"


def apx_equ(a, b):
    return abs(a - b) < PerfConfig.APX_EQUAL_THRESHOLD


def distance_between_points(p1: Collection[float], p2: Collection[float]) -> float:
    assert len(p1) == len(p2)
    return math.sqrt(sum([math.pow(x - y, 2) for x, y in zip(p1, p2)]))


class PerfBase:
    """ Base class for performance metrics"""

    def __init__(self, exp_result):
        self.exp_result = exp_result
        self.values: Union[list, dict] = []
        self.is_graphic = False

    def execute(self):
        # Deletes its copy of the results once executed, can only be run once
        if self.exp_result is not None:
            self.values = self._exec(self.exp_result)
            self.exp_result = None
            if self.is_values_based:
                self.append_mean_sd(self.values)
        return self  # Returns self to allow chaining

    @staticmethod
    def get_best_threshold_full(y, scores):
        # TODO 5 instead best F1 do best equ_odds
        """
            Finds the best threshold for the given scores given y.
            Returns tuple that gives best F1, if more than one same F1
            ties are broken on precision, if there is still a tie then broken
            on recall
        """
        assert len(y) == len(scores), 'Scores must have same length as labels'
        assert len(y) > 0, 'Expected at least one sample'
        max_ind = PerfConfig.UNDEFINED_VALUE
        max_f1 = PerfConfig.UNDEFINED_VALUE

        # Code to prevent the following warning (which kinda looks like an error and is pretty annoying as a result)
        # > /../venv/AnomalyDetection/lib/python3.10/site-packages/sklearn/metrics/_ranking.py:891:
        # > UserWarning: No positive class found in y_true, recall is set to one for all thresholds.
        all_negative_labels = True
        positive_label = [Conf.ANOM_LABEL]
        for label in y:
            if label == positive_label:
                all_negative_labels = False
                break
        if all_negative_labels:
            # Try to mimic values that would be generated but without the warning
            n = len(y)
            precisions, recalls, thresholds = [0.] * n, [1.] * n, sorted(-scores)
            precisions.append(1)
            recalls.append(0)
            # Convert to match type that would have been produced
            precisions, recalls, thresholds = np.asarray(precisions), np.asarray(recalls), np.asarray(thresholds)
        else:
            precisions, recalls, thresholds = metrics.precision_recall_curve(y, -scores)
        for i in range(len(thresholds)):
            if (precisions[i] + recalls[i]) > 0:
                # Denominator not 0, proceed
                curr_f1 = 2 * precisions[i] * recalls[i] / (precisions[i] +
                                                            recalls[i])
                if max_ind >= 0:
                    if curr_f1 > max_f1:
                        # Update F1 greater
                        max_f1 = curr_f1
                        max_ind = i
                    elif apx_equ(curr_f1, max_f1):
                        # F1 equal check precision
                        if precisions[i] > precisions[max_ind]:
                            # Update Precision Higher
                            max_f1 = curr_f1
                            max_ind = i
                        elif apx_equ(precisions[i], precisions[max_ind]):
                            # Precision also equal check recall
                            if recalls[i] > recalls[max_ind]:
                                # Update Recall Greater
                                max_f1 = curr_f1
                                max_ind = i
                else:
                    max_f1 = curr_f1
                    max_ind = i
        # TODO Fix edge case discovered that thresholds is one shorter than precision and recall lists
        return max_f1, max_ind, precisions, recalls, thresholds

    @staticmethod
    def get_best_threshold(y, scores) -> Optional[float]:
        """
            Returns only the best threshold or none if not defined
        """
        max_f1, max_ind, precisions, recalls, thresholds = \
            PerfBase.get_best_threshold_full(y, scores)
        return None if max_ind < 0 else thresholds[max_ind]

    @staticmethod
    def get_best_threshold_or_default(y, scores) -> float:
        result = PerfBase.get_best_threshold(y, scores)
        return result if result is not None else PerfBase.get_default_threshold(scores)

    @staticmethod
    def get_default_threshold(scores) -> float:
        """
        Set to value above max (effectively disable it)
        """
        return max(scores) + 1

    @abstractmethod
    def _exec(self, exp_result):
        pass

    @property
    def is_values_based(self):
        # Also allows subclasses to override and always return False e.g. Scores.
        # So it enables more than just inverting is_graphic
        return not self.is_graphic

    def headings_and_mean(self) -> Optional[Tuple[List[str], List]]:
        headings = []
        values = []

        def get_mean(vals: List):
            if not isinstance(vals, list) or len(vals) < 3:
                return "NA"
            else:
                return vals[-2]

        if isinstance(self.values, dict):
            for key in self.values.keys():
                headings.append(f'{self.name()} {key}')
                values.append(get_mean(self.values[key]))
        else:
            headings.append(self.name())
            values.append(get_mean(self.values))
        return headings, values

    def _data_to_str(self, data: Union[dict, list]) -> str:
        if isinstance(data, dict):
            result = ''
            for key in data.keys():
                result += f'\n\t\t{key}:\t{self._data_to_str(data[key])}'
            return result
        else:
            return f'{self._float_to_str(data)}'

    def name(self):
        return f'{self.__class__.__name__}'

    def __str__(self):
        return f'{self.name()} = {self._data_to_str(self.values)}'

    def to_native(self, data):
        if isinstance(data, dict):
            result = {}
            for key in data.keys():
                result[key] = self.to_native(data[key])
            return result
        if isinstance(data, Iterable):
            return [self.to_native(x) for x in data]
        return data.item() if isinstance(data, np.generic) else data

    def to_matlab(self):
        # noinspection PyTypeChecker
        return self.to_native(self.values)

    @classmethod
    def _float_to_str(cls, data: Union[dict, List[Number]]):
        if isinstance(data, dict):
            result = {}
            for key in data.keys():
                result[key] = cls._float_to_str(data[key])
            return result
        elif isinstance(data[0], dict):
            return [cls._float_to_str(x) for x in data]
        else:
            return ["%.4f" % i for i in data]

    def append_mean_sd(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                self.append_mean_sd(data[key])
        elif isinstance(data[0], Iterable):
            for i in data:
                self.append_mean_sd(i)
        else:
            # Calc mean
            sum_ = sum(data)
            mean = sum_ / len(data)

            # Calc Standard Deviation
            sum_ = 0
            for i in data:
                sum_ += (i - mean) ** 2
            sd = math.sqrt(sum_ / len(data))

            # Append to list
            data += [mean, sd]


class PerfGraphic(PerfBase):
    TIMESTAMP = None

    def __init__(self, exp_result):
        super().__init__(exp_result)
        self.is_graphic = True

    @abstractmethod
    def _exec(self, exp_result):
        pass

    @classmethod
    def viz_file_name(cls, exp, trial=None, *, ext='.png'):
        if trial is None:
            trial = ''
        else:
            trial = f' - {trial}'

        if PerfConfig.VIZ_USE_FOLDER_PER_EXP:
            folder = f'{Conf.FileLocations.RESULTS}{PerfGraphic.TIMESTAMP}/' \
                     f'{get_valid_ident(exp)}/'
            filename = f'{folder}{cls.__name__}{trial}{ext}'
        else:
            folder = f'{Conf.FileLocations.RESULTS}{PerfGraphic.TIMESTAMP}/'
            filename = f'{folder}{get_valid_ident(exp)} - {cls.__name__}' \
                       f'{trial}{ext}'

        ensure_folder_created(filename)
        return filename

    def headings_and_mean(self) -> Optional[Tuple[List[str], List]]:
        return None
