import itertools
import logging
import math
import random
from typing import List, Optional, Tuple

import numpy as np
from opylib.log import log
from sklearn.ensemble import IsolationForest

from src.enums import TCustomIFMode, TCircleRadiusReference
from src.performance.perf_base import apx_equ, distance_between_points
from src.supporting.min_max import MinMax, get_min_max_from_data


# noinspection PyPep8Naming
class CustomAlgo1:
    """
    Custom algorithm for testing purposes
    """

    def __init__(self, num_columns: int = 3, num_rows: int = 2):
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.feature_ranges: Optional[List[MinMax]] = None
        self.cell_counts: Optional[List[List[int]]] = None
        self.total_samples: Optional[int] = None
        self.cell_bounds: Optional[List[List[float]]] = None

    def fit(self, X):
        self.total_samples = X.shape[0]
        usable_features = 1 if len(X.shape) == 1 else min(X.shape[1], 2)  # TODO test on 1D dataset and more than 2D
        if usable_features == 1:
            self.num_rows = 1

        # Find ranges (min/max) for each usable feature
        self.feature_ranges = get_min_max_from_data(X, usable_features)

        # Calculate horizontal splits
        col_size = (self.feature_ranges[0].max - self.feature_ranges[0].min) / self.num_columns
        col_bounds = [self.feature_ranges[0].min + col_size * i for i in range(self.num_columns + 1)]
        col_bounds[-1] += 1e-9
        assert math.isclose(col_bounds[0], self.feature_ranges[0].min)
        assert math.isclose(col_bounds[-1], self.feature_ranges[0].max)
        self.cell_bounds = [col_bounds]

        # Calculate vertical splits if applicable
        if usable_features >= 2:
            row_size = (self.feature_ranges[1].max - self.feature_ranges[1].min) / self.num_rows
            row_bounds = [self.feature_ranges[1].min + row_size * i for i in range(self.num_rows + 1)]
            row_bounds[-1] += 1e-9
            assert math.isclose(row_bounds[0], self.feature_ranges[1].min)
            assert math.isclose(row_bounds[-1], self.feature_ranges[1].max)
            self.cell_bounds.append(row_bounds)

        self.cell_counts = [[0 for _ in range(self.num_columns)] for _ in range(self.num_rows)]
        for x in X:
            cell_pos = self.get_cell_index(x)
            assert cell_pos is not None, f"Out of bounds sample: {x}"
            col_index, row_index = cell_pos
            self.cell_counts[row_index][col_index] += 1
        assert sum([sum(x) for x in self.cell_counts]) == self.total_samples

    def score_samples(self, X):
        assert self.cell_counts is not None
        assert self.feature_ranges is not None
        assert self.total_samples is not None
        assert self.cell_bounds is not None
        result = []
        for x in X:
            sample_score = 0
            bin_index = self.get_cell_index(x)
            if bin_index is not None:
                # Only increase score if it falls in a bin
                col_index, row_index = bin_index
                sample_score = self.cell_counts[row_index][col_index] / self.total_samples
            result.append(sample_score)
        return np.asarray(result)

    def get_cell_index(self, sample) -> Optional[Tuple[int, int]]:
        assert self.cell_bounds is not None
        result: List[int] = []
        for feature_index, feature_value in enumerate(sample[:2]):
            if feature_value < self.cell_bounds[feature_index][0] \
                    or feature_value > self.cell_bounds[feature_index][-1]:
                return None
            found_value = False
            for i, bound in enumerate(self.cell_bounds[feature_index][1:]):
                if feature_value <= bound:
                    result.append(i)
                    found_value = True
                    break
            if not found_value:
                raise ValueError(
                    f"Value not out of range but still couldn't find range! min: {self.cell_bounds[feature_index][0]} "
                    + f"max: {self.cell_bounds[feature_index][-1]} feature_value: {feature_value}")
        if len(sample) == 1:
            result.append(0)  # Implicit row
        assert len(result) == 2
        col_index, row_index = result[0], result[1]
        assert 0 <= col_index < self.num_columns
        assert 0 <= row_index < self.num_rows
        return col_index, row_index


# noinspection PyPep8Naming
class CustomIF(IsolationForest):
    def __init__(self, **kwarg):
        self.spacing_absolute = kwarg.pop('spacing_absolute', 0.5)
        self.spacing_percentage = kwarg.pop('spacing_percentage', 2)  # Default 200%
        self.circle_radius_percentage = kwarg.pop('circle_radius_percentage', 0.1)
        self.circle_radius_reference = kwarg.pop('circle_radius_reference', TCircleRadiusReference.MIN_DIMENSION)
        self.points_per_feat = kwarg.pop('points_per_feat', 3)
        self.point_gen_random_state = kwarg.pop('point_gen_random_state', None)
        self.mode = kwarg.pop('mode', TCustomIFMode.SINGLE_RANDOM)
        if self.mode == TCustomIFMode.SINGLE_RANDOM:
            self.random_points_count = 1
            if kwarg.pop('random_points_count', 1) != 1:
                log("WARNING: Single random point generation mode set but random points count is not 1", logging.ERROR)
        else:
            # Use number from 0 to less than 1 for a percentage and numbers 1 and up for fixed number of points
            self.random_points_count = kwarg.pop('random_points_count', 1)
        self.additional_points = None
        super().__init__(**kwarg)

    def fit(self, *args, **kwarg):
        X = args[0]  # Get X from args
        match self.mode:
            case TCustomIFMode.SINGLE_RANDOM:
                self.additional_points = np.asarray(self.gen_points_single_random(X))
            case TCustomIFMode.SQUARE_GRID:
                self.additional_points = np.asarray(self.gen_points_square_grid(X))
            case TCustomIFMode.SINGLE_SPECIAL:
                self.additional_points = np.asarray(self.gen_points_single_special())
            case TCustomIFMode.MULTIPLE_RANDOM:
                self.additional_points = np.asarray(self.gen_points_multiple_random(X))
            case TCustomIFMode.LINE_IN_MIDDLE:
                self.additional_points = np.asarray(self.gen_points_line_in_middle(X))
            case TCustomIFMode.CIRCLE_IN_MIDDLE:
                self.additional_points = np.asarray(self.gen_points_circle_in_middle(X, False))
            case TCustomIFMode.DISK_IN_MIDDLE:
                self.additional_points = np.asarray(self.gen_points_circle_in_middle(X, True))
            case TCustomIFMode.MIN_AND_MAX:
                self.additional_points = np.asarray(self.gen_points_min_and_max(X))
            case TCustomIFMode.ALL_MAX:
                self.additional_points = np.asarray(self.gen_points_all_max(X))

        assert self.additional_points is not None, f"No points set? mode is {self.mode}"
        assert len(self.additional_points) > 0, "Expected at least 1 point to be added"
        X = np.concatenate((X, self.additional_points))  # Add on desired points
        args = tuple([X] + list(args[1:]))  # Create replacement tuple to pass on for training
        super().fit(*args, **kwarg)

    def gen_points_square_grid(self, X):
        min_maxes: List[MinMax] = [MinMax(x.min - self.spacing_absolute, x.max + self.spacing_absolute) for x in
                                   get_min_max_from_data(X)]
        dimension_values = [np.linspace(x.min, x.max, self.points_per_feat) for x in min_maxes]
        return list(itertools.product(*dimension_values))

    @staticmethod
    def gen_points_single_special():
        return [[5, 3.5]]

    def gen_points_single_random(self, X):
        assert self.random_points_count == 1, \
            "Random points count must be 1 when this mode is set should have fixed in constructor"
        return self.gen_points_multiple_random(X)

    def gen_points_multiple_random(self, X):
        self.initialize_random_state()
        min_maxes = self.get_min_max_with_spacing(X)
        count = self.calculate_random_point_count(X)

        # Generate points
        result = []
        for _ in range(count):
            result.append([random.uniform(x.min, x.max) for x in min_maxes])
        log(f"CustomIF generated {count} random points")
        return result

    def initialize_random_state(self):
        if self.point_gen_random_state is not None:
            log(f"Using seed value of {self.point_gen_random_state} for point generation (by configuration)")
            random.seed(self.point_gen_random_state)
        else:
            random_state = random.randrange(2 ** 32 - 1)  # Get a random seed for generation
            random.seed(random_state)
            log(f"Using seed value of {random_state} for point generation (randomly for reproducibility)")

    def gen_points_line_in_middle(self, X):
        self.initialize_random_state()
        min_maxes = self.get_min_max_with_spacing(X)
        count = self.calculate_random_point_count(X)

        # Generate points
        result = []
        for _ in range(count):
            result.append(
                [random.uniform(x.min, x.max) if i == 0 else ((x.min + x.max) / 2) for i, x in enumerate(min_maxes)])
        return result

    def get_min_max_with_spacing(self, X) -> List[MinMax]:
        result = self.get_min_max_raw(X)

        # Add spacing
        for min_max in result:
            width = min_max.max - min_max.min
            spacing = width * self.spacing_percentage
            min_max.min -= spacing
            min_max.max += spacing

        return result

    def calculate_random_point_count(self, X) -> int:
        assert self.random_points_count > 0
        if self.random_points_count < 1:
            result = int(len(X) * self.random_points_count)
            if result == 0:
                result += 1
            log(f"Treated {self.random_points_count=} as a percentage and set number of points "
                f"{result} of {len(X)} original points")
        else:
            result = self.random_points_count
            log(f"Using raw value for number of points to generate {result} points")

        return result

    def gen_points_circle_in_middle(self, X, filled_in: bool):
        self.initialize_random_state()
        min_maxes = self.get_min_max_raw(X)
        count = self.calculate_random_point_count(X)

        # Generate points
        result = []
        center = [((x.min + x.max) / 2) for x in min_maxes]
        widths = [x.max - x.min for x in min_maxes]
        match self.circle_radius_reference:
            case TCircleRadiusReference.MIN_DIMENSION:
                reference_width = min(widths)
            case TCircleRadiusReference.MAX_DIMENSION:
                reference_width = max(widths)
            case TCircleRadiusReference.AVERAGE_DIMENSION:
                reference_width = sum(widths) / len(widths)
            case _:
                raise ValueError(f"Unrecognized circle radius reference {self.circle_radius_reference}")
        assert reference_width > 0, \
            (f"Circle reference width {reference_width} is not positive but is required to be. "
             f"Circle Reference Type is: {self.circle_radius_reference}")
        r = self.circle_radius_percentage * reference_width
        for _ in range(count):
            if filled_in:
                point_radius = random.uniform(0, r)
            else:
                point_radius = r
            result.append(self.generate_point_circle(center, point_radius))
        return result

    @staticmethod
    def generate_point_circle(center: List[float], r: float) -> List[float]:
        n = len(center)
        if n == 1:
            # Not circle but a line or length at most 2r (diameter)
            return [center[0] + random.uniform(0, r)]
        assert n > 0
        angles = [
            random.uniform(0, math.pi)
            for _ in range(n - 2)
        ]
        angles.append(random.uniform(0, 2 * math.pi))

        result = [r * math.cos(angles[0])]
        sin_prod = 1
        for i in range(1, n - 1):
            sin_prod *= math.sin(angles[i - 1])
            result.append(r * sin_prod * math.cos(angles[i]))
        result.append(
            r * sin_prod * math.sin(angles[n - 2]))

        # Move point based on center
        for i, x in enumerate(center):
            result[i] += x

        assert apx_equ(distance_between_points(center, result), r), \
            f"This point is not on the sphere. {center=} {result=} {r=} {n=} {angles=} {sin_prod=}"
        return result

    @staticmethod
    def get_min_max_raw(X) -> List[MinMax]:
        return [MinMax(x.min, x.max) for x in get_min_max_from_data(X)]

    def gen_points_min_and_max(self, X):
        self.initialize_random_state()
        min_maxes = self.get_min_max_with_spacing(X)
        count = self.calculate_random_point_count(X)

        result = []
        for outer_idx, x in enumerate(min_maxes):
            min_point = []
            max_point = []
            for inner_idx, y in enumerate(min_maxes):
                if outer_idx == inner_idx:
                    min_point.append(x.min)
                    max_point.append(x.max)
                else:
                    avg = (x.min + x.max) / 2
                    min_point.append(avg)
                    max_point.append(avg)
            result.append(min_point)
            result.append(max_point)
        while len(result) < count:
            result = result * 2
        return result

    def gen_points_all_max(self, X):
        # No randomness used, points added are deterministic
        min_maxes = self.get_min_max_with_spacing(X)
        count = self.calculate_random_point_count(X)
        result = [[x.max for x in min_maxes]] * count
        return result
