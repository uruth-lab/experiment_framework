import math

from src.data_generator.gen_base_func import GenBaseFunc


class GenSine(GenBaseFunc):

    def __init__(self, settings: dict = None, *,
                 anom_should_rej_func: callable = None,
                 get_normal_point: callable = None):
        super().__init__(settings, anom_should_rej_func=anom_should_rej_func,
                         get_normal_point=get_normal_point)

    def name(self):
        return 'sine'

    def y_func(self, x):
        return math.sin(x)
