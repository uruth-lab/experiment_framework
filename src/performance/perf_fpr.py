import logging

from opylib.log import log
from sklearn.metrics import roc_curve

from src.performance.perf_base import PerfBase, PerfConfig, apx_equ


class FPR(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in exp_result.trials_scores:
            threshold_for_f1 = self.get_best_threshold(exp_result.dataset.y,
                                                       scores)
            if threshold_for_f1 is None:
                results.append(PerfConfig.UNDEFINED_VALUE)
            else:
                fpr, tpr, thresholds = roc_curve(exp_result.dataset.y, -scores)

                # Find matching threshold
                found = False
                for i in range(len(thresholds)):
                    if apx_equ(threshold_for_f1, thresholds[i]):
                        found = True
                        results.append(fpr[i])
                        break
                if not found:
                    # Failed to find matching threshold
                    results.append(PerfConfig.VALUE_FAIL)
                    log("FAILED TO MATCH THRESHOLD FOR FPR", logging.ERROR)
        return results
