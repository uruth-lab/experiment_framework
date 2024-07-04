from src.performance.perf_base import PerfBase


class TimeTrial(PerfBase):
    def _exec(self, exp_result):
        self.trials = []
        self.fits = []
        self.scores = []
        for time_trial in exp_result.trial_times:
            self.trials.append(time_trial.trial.as_float())
            self.fits.append(time_trial.fit.as_float())
            self.scores.append(time_trial.score.as_float())

        return {
            'Fit': self.fits,
            'Score': self.scores,
            'Total': self.trials,
        }
