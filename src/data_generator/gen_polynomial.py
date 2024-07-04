from src.data_generator.gen_base_func import GenBaseFunc


class GenPolynomial(GenBaseFunc):
    def __init__(self, settings: dict = None, *,
                 anom_should_rej_func: callable = None,
                 get_normal_point: callable = None):
        super().__init__(settings, anom_should_rej_func=anom_should_rej_func,
                         get_normal_point=get_normal_point)

    def name(self):
        return 'polynomial'

    def y_func(self, x):
        y = 0
        for i in range(len(self.coefficients)):
            y += pow(x, i) * self.coefficients[-1 - i]
        return y

    @property
    def coefficients(self):
        return self.settings['coefficients']
