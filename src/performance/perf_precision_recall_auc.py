from sklearn import metrics

from src.performance.perf_base import PerfBase


class PrecisionRecallAUC(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in exp_result.trials_scores:
            f1, max_ind, precisions, recalls, thresholds = \
                self.get_best_threshold_full(
                    exp_result.dataset.y, scores)
            results.append(metrics.auc(recalls, precisions))
        return results
