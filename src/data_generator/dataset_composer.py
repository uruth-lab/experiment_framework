import numpy as np
import scipy.io as sio

from src.config import Conf
from src.utils.misc.data_import_export import get_matlab_safe_dict
from src.utils.misc.shuffle import shuffle
from src.utils.misc.to_py_obj import str_to_class


class DatasetComposer:
    def __init__(self, id_, settings):
        self.id = id_
        self.settings = {**Conf.Defaults.DATASET_COMPOSER, **settings}
        all_gen = self.gen_anom + self.gen_normal
        self._executed_gens_normal = []
        self._executed_gens_anom = []

        def set_where_not_overridden(gen_list, param_name, value):
            for gen_ in gen_list:
                if gen_['params'].get(param_name) is None:
                    gen_['params'][param_name] = value

        # Split size evenly and set only where not already set
        # NB: May not equal size after this because of round off differences
        # or that some of the generators had their size set manually
        assert 0 <= self.anom_ratio <= 1
        total_anom_size = self.size * self.anom_ratio
        total_normal_size = self.size - total_anom_size
        each_anom_size = total_anom_size // len(self.gen_anom)
        each_normal_size = total_normal_size // len(self.gen_normal)
        set_where_not_overridden(self.gen_anom, 'size', each_anom_size)
        set_where_not_overridden(self.gen_normal, 'size', each_normal_size)

        # Set number of features for dataset that have not overridden the value
        set_where_not_overridden(
            all_gen, 'feats_informative', self.feats_informative)
        set_where_not_overridden(
            all_gen, 'feats_irrelevant', self.feats_irrelevant)

        # Ensure all transformations set are the correct dimensions
        n = self.feats_informative + self.feats_irrelevant
        for gen in all_gen:
            transform = gen.get('transform')
            if transform is not None and np.asarray(transform).shape != (n, n):
                raise Exception(
                    f'Expected {(n, n)} for transform but got '
                    f'{np.asarray(transform).shape} for {gen}')

    def generate(self):
        # Generate Data points
        self._execute_gens(self.gen_normal, is_normal=True)
        self._execute_gens(self.gen_anom, is_normal=False)

        # Gather up generated points, shuffle and save
        self._gather_shuffle_and_save()

    def gen_auto_output_filename(self):
        def get_names(gen_list):
            result = None
            for gen in gen_list:
                if result is None:
                    result = str(gen['gen'])
                else:
                    result += f'_{gen["gen"]}'
            return result

        return f'{self.id}' \
               f'_Norm({get_names(self.gen_normal)})' \
               f'_Anom({get_names(self.gen_anom)})'

    def __str__(self):
        return f'{self.id} - {self.output_filename}'

    def __len__(self):
        return sum(len(x) for x in self._executed_gens_normal) + sum(
            len(x) for x in self._executed_gens_anom)

    def _anom_should_rej_func(self, point):
        return any([x.should_rej(point) for x in self._executed_gens_normal])

    def _execute_gens(self, gen_list: list, *, is_normal: bool):
        for gen in gen_list:
            constructor = str_to_class(gen['gen'].value)
            anom_should_rej_func = None if is_normal else \
                self._anom_should_rej_func

            get_normal_point = None
            if gen.get('feeder') is not None:
                # TODO 4 Implement optional parameter for feeder generator
                # Create a 'feeder' generator and assign generate point
                # function to variable
                raise NotImplemented('Feeder not implemented yet')

            generator = constructor(gen['params'],
                                    anom_should_rej_func=anom_should_rej_func,
                                    get_normal_point=get_normal_point)
            generator.generate()

            transformation = gen.get('transform')
            if transformation is not None:
                self._do_transformation(generator.X, transformation)

            if is_normal:
                self._executed_gens_normal.append(generator)
            else:
                self._executed_gens_anom.append(generator)

    # noinspection PyPep8Naming
    def _gather_shuffle_and_save(self):
        X = []
        y = []
        groups = []

        def add_list(gen_list, y_val):
            nonlocal X, y, groups
            for generator in gen_list:
                X += generator.X
                y += [y_val] * len(generator.X)
                groups += [generator.group] * len(generator.X)

        # Collect Values
        add_list(self._executed_gens_normal, [0])
        add_list(self._executed_gens_anom, [1])

        # Shuffle
        if self.should_shuffle:
            X, y, groups = shuffle(X, y, groups)

        # Save
        sio.savemat(
            Conf.FileLocations.DATASETS.substitute(name=self.output_filename),
            {
                'X': X,
                'y': y,
                'groups': groups,
                'settings': get_matlab_safe_dict(self.settings),

                # WARNING: Some individual generators may have overridden this
                # value but this is expected to be rare
                'display_note': f'({self.feats_informative}, '
                                f'{self.feats_irrelevant})'
            })

    @staticmethod
    def _do_transformation(points, transformation):
        transformation = np.asarray(transformation)
        for i in range(len(points)):
            point = np.asarray(points[i])
            # TODO Check if only 2D transformations are supported
            point = point.reshape(2, 1)
            point = transformation @ point
            points[i] = list(point)

    @property
    def output_filename(self):
        return self.settings['output_filename'] if self.settings.get(
            'output_filename') is not None else self.gen_auto_output_filename()

    @property
    def size(self):
        return self.settings['size']

    @property
    def anom_ratio(self):
        return self.settings['anom_ratio']

    @property
    def should_shuffle(self):
        return self.settings['should_shuffle']

    @property
    def feats_informative(self):
        return self.settings['feats_informative']

    @property
    def feats_irrelevant(self):
        return self.settings['feats_irrelevant']

    @property
    def gen_normal(self):
        return self.settings['gen_normal']

    @property
    def gen_anom(self):
        return self.settings['gen_anom']
