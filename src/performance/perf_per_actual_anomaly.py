from src.performance.perf_base import PerfBase


class PercentActualAnomaly(PerfBase):
    def _exec(self, exp_result):
        anomaly_count = 0
        for val in exp_result.dataset.y:
            if val == 1:
                anomaly_count += 1
        results = [anomaly_count / len(exp_result.dataset.y)]
        return results
