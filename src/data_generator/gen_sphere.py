import math
import random
from typing import Iterable

from src.data_generator.gen_base import GenBase


class GenSphere(GenBase):
    """Normal on Sphere"""

    def __init__(self, settings: dict = None, *,
                 anom_should_rej_func: callable = None,
                 get_normal_point: callable = None):
        super().__init__(settings, anom_should_rej_func=anom_should_rej_func,
                         get_normal_point=get_normal_point)

    def name(self):
        return 'sphere'

    def get_point(self):
        n = self.feats_informative
        angles = [
            random.uniform(0, math.pi)
            if self.fixed_angles.get(i) is None else
            self.fixed_angles[i]
            for i in range(n - 2)
        ]
        angles.append(
            random.uniform(0, 2 * math.pi)
            if self.fixed_angles.get(n - 2) is None else
            self.fixed_angles[n - 2])
        r = self.radius
        result = [r * math.cos(angles[0])]
        sin_prod = 1
        for i in range(1, n - 1):
            sin_prod *= math.sin(angles[i - 1])
            result.append(r * sin_prod * math.cos(angles[i]))
        result.append(
            r * sin_prod * math.sin(angles[n - 2]))
        noise = self.get_noise()
        for i, val in enumerate(noise):
            result[i] += val
        return result

    def should_rej(self, point: Iterable[float]) -> bool:
        point_distance = math.sqrt(sum(i ** 2 for i in point))
        min_allowed = self.radius - self.buffer_for_noise
        max_allowed = self.radius + self.buffer_for_noise
        return point_distance <= min_allowed or point_distance > max_allowed

    @property
    def radius(self):
        return self.settings['radius']

    @property
    def fixed_angles(self):
        return self.settings['fixed_angles']
