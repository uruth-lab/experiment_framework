import math
from typing import Iterable

from scipy.stats import norm

from src.data_generator.gen_base import GenBase


class GenGauss(GenBase):

    def __init__(self, settings: dict = None, *,
                 anom_should_rej_func: callable = None,
                 get_normal_point: callable = None):
        super().__init__(settings, anom_should_rej_func=anom_should_rej_func,
                         get_normal_point=get_normal_point)
        self.repeat_last_elem_to_dimen(self.std_dev)

    def name(self):
        return 'gauss'

    def get_point(self):
        return norm.rvs(scale=self.std_dev)

    def should_rej(self, point: Iterable[float]) -> bool:
        point_distance = math.sqrt(sum(i ** 2 for i in point))

        # TODO? Find better way to get distance (average of sd may not be best)
        allowed_min_distance = sum(self.std_dev) / len(
            self.std_dev) * self.rej_sd_coefficient + self.buffer_for_noise

        return point_distance < allowed_min_distance

    @property
    def std_dev(self):
        return self.settings['std_dev']
