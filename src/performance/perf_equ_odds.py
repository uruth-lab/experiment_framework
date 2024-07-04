from typing import List

from src.config import Conf
from src.performance.perf_base import PerfBase, PerfConfig
from src.supporting.data_handling import Dataset


class EqualizedOdds(PerfBase):
    def __init__(self, exp_result):
        super().__init__(exp_result)
        
        # Sum of differences between fpr and fnr
        self.diff_sum = []

        # Difference Between False positive rate from mean
        self.diff_fpr = []

        # Difference Between False negative rate from mean
        self.diff_fnr = []

        # Raw rates from each iteration then each group
        self.raw = []

    def _exec(self, exp_result):
        for scores in exp_result.trials_scores:
            f1, max_ind, precisions, recalls, thresholds = \
                self.get_best_threshold_full(
                    exp_result.dataset.y, scores)
            if f1 == PerfConfig.UNDEFINED_VALUE:
                self.diff_sum.append(PerfConfig.UNDEFINED_VALUE)
                self.diff_fpr.append(PerfConfig.UNDEFINED_VALUE)
                self.diff_fnr.append(PerfConfig.UNDEFINED_VALUE)
                self.raw.append(PerfConfig.UNDEFINED_VALUE)
            else:
                self.calc_equalized_odds(
                    self.exp_result.dataset, thresholds[max_ind], scores)
        return {
            'SUM': self.diff_sum,
            'FPR': self.diff_fpr,
            'FNR': self.diff_fnr,
            'RAW': self.raw,
        }

    def calc_equalized_odds(self, dataset: Dataset, threshold: float,
                            scores: List[float]):
        # Suffix Explanations in Variable Names
        # p - Positives
        # n - Negatives
        # fp - False Positive
        # fn - False Negative

        p = [0] * dataset.group_count
        n = [0] * dataset.group_count
        fp = [0] * dataset.group_count
        fn = [0] * dataset.group_count
        fpr = [0] * dataset.group_count
        fnr = [0] * dataset.group_count

        for i in range(len(dataset.X)):
            if dataset.y[i] == Conf.ANOM_LABEL:
                p[dataset.groups[i]] += 1
            else:
                n[dataset.groups[i]] += 1

            pred = 1 if -scores[i] >= threshold else 0
            if pred == dataset.y[i]:
                pass  # Nothing done if correct (matches ground truth)
            else:
                if pred == 1:
                    fp[dataset.groups[i]] += 1
                else:
                    fn[dataset.groups[i]] += 1

        assert (sum(p) + sum(n)) == len(dataset.y)

        for i in range(dataset.group_count):
            fpr[i] = 0 if n[i] == 0 else fp[i] / n[i]
            fnr[i] = 0 if p[i] == 0 else fn[i] / p[i]
            assert 0 <= fpr[i] <= 1
            assert 0 <= fnr[i] <= 1

        fpr_mean = sum(fpr) / dataset.group_count
        fnr_mean = sum(fnr) / dataset.group_count

        diff_fnr = 0
        diff_fpr = 0
        for i in range(dataset.group_count):
            diff_fnr += abs(fnr_mean - fnr[i])
            diff_fpr += abs(fpr_mean - fpr[i])

        self.diff_fpr.append(diff_fpr)
        self.diff_fnr.append(diff_fnr)
        self.diff_sum.append(diff_fpr + diff_fpr)
        self.raw.append({
            'fnr': fnr,
            'fpr': fpr,
        })
