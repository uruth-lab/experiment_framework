import math
import random
from abc import abstractmethod
from typing import Iterable

from src.data_generator.gen_base import GenBase


class GenBaseFunc(GenBase):
    """
    Base for 2D generators that produce functions
    """

    @abstractmethod
    def y_func(self, x):
        pass

    def get_point(self):
        x = random.uniform(self.x_min_max[0], self.x_min_max[1])
        y = self.y_func(x)
        result = [x, y] if not self.invert_xy else [y, x]
        if self.dimens > 2:
            # If higher than 2 dimens then fill others with 0
            result.append(0)
            self.repeat_last_elem_to_dimen(result)
        return result

    # noinspection PyUnresolvedReferences
    def should_rej(self, point: Iterable[float]) -> bool:
        # TODO : 4 Test against heavy uniform anom
        x_ind = 0 if not self.invert_xy else 1
        y_ind = (x_ind + 1) % 2
        x1 = point[x_ind]
        x2 = x1
        y1 = point[y_ind]
        y2 = self.y_func(x2)
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance < self.buffer_for_noise

    @property
    def x_min_max(self):
        return self.settings['x_min_max']

    @property
    def invert_xy(self):
        return self.settings['invert_xy']
