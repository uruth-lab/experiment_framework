from src.performance.perf_base import PerfBase, PerfConfig


class PercentPred(PerfBase):
    def _exec(self, exp_result):
        results = []
        for scores in exp_result.trials_scores:
            threshold = self.get_best_threshold(exp_result.dataset.y, scores)
            if threshold is None:
                results.append(PerfConfig.UNDEFINED_VALUE)
            else:
                anomaly_count = 0
                for score in scores:
                    if -score >= threshold:
                        anomaly_count += 1
                results.append(anomaly_count / len(scores))
        return results
