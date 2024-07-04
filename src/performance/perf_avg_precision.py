from sklearn.metrics import average_precision_score

from src.performance.perf_base import PerfBase


class AvgPrecision(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in exp_result.trials_scores:
            results.append(
                average_precision_score(exp_result.dataset.y, -scores))
        return results
