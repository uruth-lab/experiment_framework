import random
from typing import Iterable

from src.data_generator.gen_base import GenBase


class GenUniform(GenBase):
    def __init__(self, settings: dict = None, *,
                 anom_should_rej_func: callable = None,
                 get_normal_point: callable = None):
        super().__init__(settings, anom_should_rej_func=anom_should_rej_func,
                         get_normal_point=get_normal_point)
        self.repeat_last_elem_to_dimen(self.feat_range)

    def get_point(self):
        result = []
        for i in range(self.dimens):
            result.append(
                random.uniform(
                    self.feat_range[i][0],
                    self.feat_range[i][1]))
        return result

    def name(self):
        return 'uniform'

    # noinspection PyUnresolvedReferences
    def should_rej(self, point: Iterable[float]) -> bool:
        for i in range(self.dimens):
            min_ = self.feat_range[i][0] - self.buffer_for_noise
            max_ = self.feat_range[i][1] + self.buffer_for_noise
            if min_ < point[i] < max_:
                return True
        return False

    @property
    def feat_range(self):
        return self.settings['feat_range']
