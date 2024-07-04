from typing import List, Optional

from opylib.stopwatch import StopWatch
from sklearn.neighbors import LocalOutlierFactor

from src.performance.perf_f1 import F1
from src.supporting.no_depend import get_valid_ident
from src.utils.misc.data_import_export import save_yaml


class AlgorithmContainer:
    def __init__(self, constructor, args, score, nick_name: Optional[str], *, needs_transposed=False,
                 is_deterministic=False):
        self.constructor = constructor
        self.nick_name: Optional[str] = nick_name
        self.args = args
        self.needs_transposed = needs_transposed
        self.is_deterministic = is_deterministic
        self.score = score

    def __str__(self):
        match f'{self.constructor.__name__}':
            case "iForest":
                result = "EIF"
            case "LocalOutlierFactor":
                result = "LofDefault" if self.is_lof_without_novelty() else "LofNovelty"
            case other:
                result = other
        result += f',{"" if self.nick_name is None else self.nick_name}'
        return result

    def is_lof_without_novelty(self):
        return LocalOutlierFactor.__name__ == self.constructor.__name__ and not self.args.get('novelty')


class Experiment:
    def __init__(self, id_, algorithm, dataset, performance, trials, performance_config):
        self.id = id_
        self.algorithm = algorithm
        self.dataset = dataset
        self.performance = performance
        self.trials = trials
        self.classifiers = []
        self.performance_config = performance_config

    def add_clf(self, clf):
        self.classifiers.append(clf)

    def __str__(self):
        return f'{self.id}-{self.algorithm}, {self.dataset}'


class TrialTimes:
    def __init__(self, trial: StopWatch, fit: StopWatch, score: StopWatch):
        self.trial = trial
        self.fit = fit
        self.score = score


class ExperimentResult:
    def __init__(self, exp: Experiment, trials_scores: List[TrialTimes],
                 exp_time: StopWatch,
                 trial_times, dataset):
        self.exp = exp
        self.trials_scores = trials_scores
        self.exp_time = exp_time
        self.trial_times = trial_times
        self.dataset = dataset

    def __str__(self):
        return f'{self.exp}{self.dataset.display_note}'

    def classifiers_and_scores(self):
        assert len(self.exp.classifiers) == self.trials_count
        return zip(self.exp.classifiers, self.trials_scores)

    @property
    def trials_count(self):
        return len(self.trials_scores)


class ExperimentPerformance:
    def __init__(self, exp_result: ExperimentResult):
        self.exp = exp_result.exp
        self.perfs = []
        sw_pc = StopWatch(f'{exp_result} Performance Check')
        for perf_constructor in exp_result.exp.performance:
            sw_perf = StopWatch(
                f'{perf_constructor.__name__} - {exp_result}')
            self.perfs.append(perf_constructor(exp_result).execute())
            sw_perf.end()
        sw_pc.end()

    def __str__(self):
        """
        Returns a string of all the performances except for scores and graphics
        :return:
        """
        result = f'{self.exp}\n'
        for perf in self.perfs:
            if perf.is_values_based:
                result += f'\t{perf}\n'
        return result

    def to_dict(self):
        result = {}
        for perf in self.perfs:
            if not perf.is_graphic:
                result[
                    get_valid_ident(
                        perf.__class__.__name__)] = perf.to_matlab()
        return result


class ExperimentPerformances:
    def __init__(self):
        self.data = []

    def add(self, exp_result: ExperimentResult):
        self.data.append(ExperimentPerformance(exp_result))

    def __str__(self):
        result = ''
        for elem in self.data:
            result += f'{elem}\n'
        return result

    def save_to_file(self, filename: str):
        save_yaml(self.to_dict(), filename)

    def to_dict(self):
        result = {}
        for elem in self.data:
            result[get_valid_ident(elem.exp)] = elem.to_dict()
        return result

    def add_summary(self):
        """
        Done quickly only able to be run once, adds an object to end of data
        to give a summary of f1 scores and open space detection
        :return:
        """
        self.data.append(PerformancesSummaryF1(self.data))
        # TODO 1: Add summary for open space detection

    def get_results_for_table(self) -> List[List]:
        exp_perfs = self.data[:-1]  # Drop last to avoid summary
        result = [["Experiment"]]
        if len(exp_perfs) == 0:
            return result
        headings = result[-1]
        exp_map = {}

        # Add row for each experiment
        for exp_perf in exp_perfs:
            result.append([f"{exp_perf.exp}"])
            exp_map[exp_perf] = result[-1]

        # Create column for each performance that is not graphical
        # NB: Only performances in the first experiment get columns created for them
        for first_perf in exp_perfs[0].perfs:
            gen_result = first_perf.headings_and_mean()
            if gen_result is None:
                continue
            first_heading = gen_result[0]
            first_perf_name = first_perf.name()
            headings += first_heading
            for exp_perf in exp_perfs:
                # Use values from first matching name (Assumes: Names are unique (duplicates ignored))
                for perf in exp_perf.perfs:
                    if perf.name() == first_perf_name:
                        this_heading, values = perf.headings_and_mean()
                        assert this_heading == first_heading, "Expected headings to match because name matched"
                        exp_map[exp_perf] += values
                        break
                else:
                    exp_map[exp_perf] += ["NA"] * len(first_heading)
        return result


class PerformancesSummaryF1:
    def __init__(self, exp_perfs: List[ExperimentPerformance]):
        """
        Extracts f1 scores from each performance and builds a dict
        :param exp_perfs: list of performance objects to extract from
        """

        # Stored here to be compatible with a normal performance object
        self.exp = 'Summary of F1 scores'

        self.data = {}
        for exp_perf in exp_perfs:
            f1 = None
            for x in exp_perf.perfs:
                if isinstance(x, F1):
                    f1 = x
                    break
            if f1 is not None:
                self.data[get_valid_ident(exp_perf.exp)] = f1

    def __str__(self):
        result = f'{self.exp}\n'
        for name, value in self.data.items():
            result += f'\t{name}\n\t\t{value}\n'
        return result

    def to_dict(self):
        result = {}
        for name, value in self.data.items():
            result[name] = value.to_matlab()
        return result
