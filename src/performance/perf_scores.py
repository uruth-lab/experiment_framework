from typing import Optional, Tuple, List

import numpy as np

from src.custom_algo.algorithms import CustomIF
from src.performance.perf_base import PerfBase
from src.supporting.data_organization import ExperimentResult


def scores_iter(exp_result: ExperimentResult):
    """
    Applies corrections that may be needed for certain classifiers
    :return: An iterator over the corrected scores for all classifiers
    """
    for clf, scores in exp_result.classifiers_and_scores():
        if isinstance(clf, CustomIF):
            additional_scores = exp_result.exp.algorithm.score(clf, clf.additional_points)
            scores = np.concatenate((scores, additional_scores))
        yield scores


class Scores(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in scores_iter(exp_result):
            results.append(scores)
        return results

    def __str__(self):
        return f'{self.__class__.__name__} = ' \
               f'{[self._float_to_str(val) for val in self.values]}'

    def append_mean_sd(self, data):
        pass  # Do not add to scores

    def is_values_based(self):
        return False

    def headings_and_mean(self) -> Optional[Tuple[List[str], List]]:
        return None


class ScoresMin(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in scores_iter(exp_result):
            results.append(min(scores))
        return results


class ScoresMax(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in scores_iter(exp_result):
            results.append(max(scores))
        return results
