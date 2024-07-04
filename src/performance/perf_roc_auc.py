from sklearn import metrics

from src.performance.perf_base import PerfBase


# noinspection PyPep8Naming
class ROC_AUC(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in exp_result.trials_scores:
            fpr, tpr, thresholds = metrics.roc_curve(exp_result.dataset.y,
                                                     -scores)
            results.append(metrics.auc(fpr, tpr))
        return results
