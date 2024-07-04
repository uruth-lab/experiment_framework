import inspect
import random
from abc import abstractmethod
from typing import Iterable

from scipy.stats import norm

from src.config import Conf
from src.enums import TDatasetGen


class GenBase:
    """
    Base of all generators
    """

    def __init__(self, settings: dict = None, *,
                 anom_should_rej_func: callable = None,
                 get_normal_point: callable = None):
        """
        Initialize the class
        :param settings: The values to use to override the defaults
        :param anom_should_rej_func: Meant to be used to do rejection
            sampling for anomalies. If set each point is passed to this
            function to see if it is rejected as too near to the normal points.
        :param get_normal_point: Meant to be used by anomalies that are
            based on normal points. To allow them access to get normal points.
        """
        self.get_normal_point = get_normal_point
        self.anom_should_rej_func = anom_should_rej_func
        self.X = []
        if settings is None:
            settings = {}
        self.settings = self.get_settings_plus_defaults(settings)
        self.repeat_last_elem_to_dimen(self.noise_sd,
                                       should_include_irrelevant=True)
        self.repeat_last_elem_to_dimen(self.offsets)
        self.rejected = 0

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_point(self):
        pass

    @abstractmethod
    def should_rej(self, point: Iterable[float]) -> bool:
        """
        Determines if a point lies too near to the normal points and should
            be rejected
        :param point: The point to test
        :return: True if the point should be reject else False
        """
        # - Can add default implementation to get the average distance between
        #   points and store it (remove stored value if more points are added)
        # - Then check the min distance from the point passed and if less than
        #   average reject
        # - Not done because it would be slow to check the distance to each
        #   point to be able to add each anomaly and may never get used
        return False

    def generate(self):
        max_rejects = self.size * self.anom_max_rej_ratio
        while len(self) < self.size:
            point = self.get_point()
            if self.feats_irrelevant > 0:
                point += self.get_irrelevant_feat_values()
            if self.should_add_noise:
                noise = self.get_noise()
                for i in range(len(point)):
                    point[i] += noise[i]
            for i, offset in enumerate(self.offsets):
                point[i] += offset
            if self.anom_should_rej_func is not None:
                if self.anom_should_rej_func(point):
                    self.rejected += 1
                    if self.rejected > max_rejects:
                        raise Exception(
                            f'Rejected points Exceeded Allowed ('
                            f'{max_rejects}). Generator "{self}" generated '
                            f'{len(self)} of {self.size} points')
                    continue
            self.X.append(point)

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return self.name()

    def get_irrelevant_feat_values(self):
        return [random.uniform(self.irrelevant_feat_range[0],
                               self.irrelevant_feat_range[1])
                for _ in range(self.feats_irrelevant)]

    def get_settings_defaults(self):
        inheritance_list = list(inspect.getmro(self.__class__))
        inheritance_list.reverse()  # Reverse order to highest first
        inheritance_list = inheritance_list[1:]  # Remove Object from Hierarchy
        result = {}
        for class_ in inheritance_list:
            enum_class = TDatasetGen(f'{class_.__module__}.{class_.__name__}')
            default_for_class = \
                Conf.Defaults.DATASET_GEN_SETTING.get(enum_class)
            if default_for_class is not None:
                result = {**result, **default_for_class}
        return result

    def get_settings_plus_defaults(self, settings):
        """
        Combines default values with the settings passed. Settings passed
        takes precedence
        :param settings: The settings passed here take precedence over defaults
        :return: The combined settings
        """
        return {**self.get_settings_defaults(), **settings}

    def repeat_last_elem_to_dimen(self, values, *,
                                  should_include_irrelevant=False):
        """
        Ensures the number of elements in values is at least self.dimens or
            self.dimens + self.feats_irrelevant if should_include_irrelevant
        Requires: len(values) is greater than 0
        :param values: The list to be extended in length
        :param should_include_irrelevant: toggles if self.feats_irrelevant
            is included in the total number of elements needed
        """
        assert len(values) > 0
        total_req = self.dimens if not should_include_irrelevant else \
            self.dimens + self.feats_irrelevant
        diff = total_req - len(values)
        if diff > 0:
            dup_value = values[-1]
            for _ in range(diff):
                values.append(dup_value)

    def get_noise(self):
        result = norm.rvs(scale=self.noise_sd)
        if self.max_abs_noise >= 0:
            for i in range(len(result)):
                if abs(result[i]) > self.max_abs_noise:
                    result[i] = (
                        self.max_abs_noise
                        if result > 0 else
                        -self.max_abs_noise)
        return result

    @property
    def size(self) -> int:
        return self.settings['size']

    @property
    def feats_informative(self):
        return self.settings['feats_informative']

    @property
    def feats_irrelevant(self):
        return self.settings['feats_irrelevant']

    @property
    def irrelevant_feat_range(self):
        return self.settings['irrelevant_feat_range']

    @property
    def should_add_noise(self):
        return self.settings['should_add_noise']

    @property
    def noise_sd(self):
        return self.settings['noise_sd']

    @property
    def max_abs_noise(self):
        return self.settings['max_abs_noise']

    @property
    def rej_sd_coefficient(self):
        return self.settings['rej_sd_coefficient']

    @property
    def anom_max_rej_ratio(self):
        return self.settings['anom_max_rej_ratio']

    @property
    def offsets(self):
        return self.settings['offsets']

    @property
    def group(self):
        return self.settings['group']

    @property
    def buffer_for_noise(self):
        # TODO : ? assess how useful a additional constant buffer would be for
        #   when the noise is off

        # TODO ? Is average of SD best value to use
        avg_sd = sum(self.noise_sd) / len(self.noise_sd)
        return 0 if not self.should_add_noise else \
            avg_sd * self.rej_sd_coefficient

    @property
    def dimens(self):
        return self.feats_informative
