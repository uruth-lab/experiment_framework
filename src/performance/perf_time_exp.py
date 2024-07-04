from src.performance.perf_base import PerfBase


class TimeExperiment(PerfBase):
    def _exec(self, exp_result):
        return [exp_result.exp_time.as_float()]
