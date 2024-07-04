from collections.abc import Iterable

import numpy as np
import scipy.io as sio
from opylib.log import log

from src.config import Conf
from src.supporting.data_organization import Experiment


class Dataset:
    def __init__(self, filename: str):
        mat_data = sio.loadmat(filename)
        # noinspection PyPep8Naming
        self.X = mat_data['X']
        if self.X.dtype.kind != 'f':
            # Superseded code that checked for u8's
            # This was needed for EIF because it can only work with double
            # And data loads as integer types if there are not factional parts independent of data type in file
            self.X = self.X.astype(float)
        if Conf.Projection.ENABLED:
            log(f"Data shape before projection: {self.X.shape}")
            self.X = self.project_data(self.X)
        log(f"Data shape: {self.X.shape}")
        self.y = mat_data['y']
        # noinspection PyPep8Naming
        self.X_trans = np.transpose(self.X)

        self.groups = mat_data.get('groups')
        if self.groups is None:
            # Set everything to not protected
            self.groups = [0] * len(self.X)

        if isinstance(self.groups[0], Iterable):
            # Matlab format converts to 2d array, converting back to 1d
            self.groups = self.groups[0]

        # Uses [G01A01]
        self.group_count = max(self.groups) + 1

        self.display_note = mat_data.get('display_note')
        if self.display_note is None:
            self.display_note = ''
        else:
            self.display_note = self.display_note[0]

    _data_dict = {}

    @classmethod
    def load_data(cls, exp: Experiment):
        filename = Conf.FileLocations.DATASETS.substitute(name=exp.dataset)
        result = cls._data_dict.get(filename)
        if result is None:
            cls._data_dict[filename] = Dataset(filename)
            result = cls._data_dict[filename]
        return result

    @staticmethod
    def project_data(data):
        rng = np.random.default_rng()
        last_dim = data.shape[-1]
        transform = rng.standard_normal((last_dim, Conf.Projection.DIMENSION))
        return data @ transform
