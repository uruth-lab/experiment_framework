import logging
import os
import random
import tracemalloc
from copy import deepcopy

from eif import iForest
from opylib.log import log, log_exception
from opylib.main_runner import main_runner
from opylib.stopwatch import StopWatch
from sklearn.ensemble import IsolationForest

from src.config import Conf
from src.custom_algo.algorithms import CustomIF
from src.enums import TGroupDatasets, TGroupPerformance, TDataset
from src.performance.perf_base import PerfGraphic
from src.pid_forest.pid_forest import PIDForest
from src.supporting.data_handling import Dataset
from src.supporting.data_organization import (AlgorithmContainer, Experiment,
                                              ExperimentPerformances,
                                              ExperimentResult, TrialTimes)
from src.supporting.no_depend import (get_run_timestamp)
from src.utils.misc.data_import_export import save_yaml, latex_convert_to_table, csv_write_rows
from src.utils.misc.files_folders import ensure_folder_created
from src.utils.misc.to_py_obj import getattr_multi_level, str_to_class


def run_experiment(exp: Experiment):
    sw_exp = StopWatch(f'********** Experiment: {exp} - ')
    dataset = Dataset.load_data(exp)
    data = dataset.X_trans if exp.algorithm.needs_transposed else dataset.X
    scores = []
    trial_times = []
    for i in range(exp.trials):
        sw_trial = StopWatch(f'***** {exp} Trial - {i + 1}')
        sw_fit = StopWatch(f'Fit')
        if exp.algorithm.constructor == iForest:
            # This is EIF
            args = deepcopy(
                exp.algorithm.args)  # Needs to be copied cuz we modify it (But needs to be modified per experiment)
            if args['ExtensionLevel'] is None:
                args['ExtensionLevel'] = data.shape[1] - 1
            args['sample_size'] = min(args['sample_size'], dataset.y.size)
            clf = exp.algorithm.constructor(
                **{'X': data, **args})
        else:
            args = exp.algorithm.args

            # For Isolation Forest and CustomIF log the seed set or set a seed and log that
            if exp.algorithm.constructor == IsolationForest or exp.algorithm.constructor == CustomIF:
                random_state = args.get('random_state')
                if random_state is not None:
                    log(f"{exp} (trial {i + 1}) random_state is set (by configuration) to {random_state}")
                else:
                    random_state = random.randrange(2 ** 32 - 1)  # Get a random seed for the experiment
                    args = deepcopy(args)  # Needs to be copied cuz we modify it and needs to diff for each experiment
                    args['random_state'] = random_state
                    log(f"{exp} (trial {i + 1}) random_state is set (randomly for reproducibility) to {random_state}")

            clf = exp.algorithm.constructor(**args)
            if isinstance(clf, PIDForest):
                # TODO 5 Fix max samples being locked to lower value once a smaller dataset gets used
                #  fix could be a simple as after the fit restore the previous value of max_samples
                clf.max_samples = min(clf.max_samples, dataset.y.size)

            clf.fit(data)
        sw_fit.end()
        exp.add_clf(clf)
        sw_score = StopWatch(f'Score')
        scores.append(exp.algorithm.score(clf, data))
        sw_score.end()
        sw_trial.end()
        trial_times.append(TrialTimes(sw_trial, sw_fit, sw_score))
    sw_exp.end()
    return ExperimentResult(exp, scores, sw_exp, trial_times, dataset)


def generate_experiment_list(input_experiments):
    id_ = 1
    generated_experiments = []
    all_settings = []
    for experiment in input_experiments:

        # Include defaults, which are then overwritten by values set in experiment
        experiment = {**Conf.Defaults.EXPERIMENT, **experiment}

        # Include default algorithm parameters
        algorithms = experiment['algorithms']
        for algorithm_pair in algorithms:
            algorithm_pair['params'] = \
                {**Conf.Defaults.ALGORITHM_PARAMETERS[algorithm_pair['algo']],
                 **algorithm_pair['params']}

        # Get dataset list
        datasets = experiment['datasets']
        if isinstance(datasets, TGroupDatasets):
            datasets = getattr_multi_level(Conf, datasets.value)
        datasets_before_conversion_to_filenames = datasets
        datasets = []
        for dataset in datasets_before_conversion_to_filenames:
            if isinstance(dataset, TDataset):
                datasets.append(dataset.value)
            elif isinstance(dataset, str):
                datasets.append(dataset)
            else:
                raise f"Unsupported dataset type: Expected TDataset or str but got {type(dataset)}"

        # Get performance list
        performance = experiment['performance']
        if isinstance(performance, TGroupPerformance):
            performance = getattr_multi_level(Conf, performance.value)
        performance = [str_to_class(x.value) for x in performance]

        # Get performance configurations
        performance_config = experiment['performance_config']

        for dataset in datasets:
            for algorithm in algorithms:
                algo = algorithm['algo']
                algorithm_container = AlgorithmContainer(
                    str_to_class(algo.value),
                    algorithm['params'],
                    str_to_class(Conf.Algorithms.SCORING[algo].value),
                    algorithm.get('name'),  # If not set will be None and will just be empty in printed name
                    needs_transposed=algo in Conf.Algorithms.REQ_TRANSPOSE,
                    is_deterministic=algo in Conf.Algorithms.DETERMINISTIC
                )
                generated_experiments.append(
                    Experiment(id_, algorithm_container, dataset, performance,
                               experiment['trials'], performance_config))
                all_settings.append({
                    'name': f'{generated_experiments[-1]}',
                    'values': {
                        'ID': id_,
                        'algorithm': algo.name,
                        'algo_params': algorithm['params'],
                        'dataset': dataset,
                        'performance': performance,
                        'trials': experiment['trials'],
                        'performance_config': performance_config,
                    },
                })
                id_ += 1
                # TODO ? save config to all
    return generated_experiments, all_settings


def save_setting(settings, fn: str):
    save_yaml(settings, fn)


def generate_results_latex_table(exp_performances: ExperimentPerformances, run_timestamp: str):
    fn = Conf.FileLocations.RESULTS_TABLE_LATEX.substitute(name=f'{run_timestamp}')
    data = exp_performances.get_results_for_table()
    output = latex_convert_to_table(data)
    save_report_to_file(output, "Results as Latex Table", fn)


def generate_results_csv(exp_performances: ExperimentPerformances, run_timestamp: str):
    fn = Conf.FileLocations.RESULTS_CSV.substitute(name=f'{run_timestamp}')
    data = exp_performances.get_results_for_table()
    csv_write_rows(data, fn)
    log(f"Report: Results as CSV saved at '{fn}'")


def save_report_to_file(report_content: str, report_name: str, fn: str):
    ensure_folder_created(fn)
    with open(fn, 'w') as f:
        f.write(report_content)
    log(f"Report: {report_name} saved at '{fn}'")


def main(run_timestamp=None, experiments=Conf.EXPERIMENTS):
    """
    Runs the passed experiments
    :param experiments: The experiments to be run
    :param run_timestamp: Allows for passing in the timestamp in use in the event of chaining with generation or some
                          other step
    """
    if Conf.MEMORY_TRACING:
        tracemalloc.start()
    if run_timestamp is None:
        run_timestamp = get_run_timestamp()
    PerfGraphic.TIMESTAMP = run_timestamp
    log(f"Called from main_.py {run_timestamp=} (To make it easier to associate with other output)")

    # Ensure output folder does not exist (To prevent overwriting old results)
    output_folder = f'{Conf.FileLocations.RESULTS}{run_timestamp}'
    if os.path.exists(output_folder):
        raise Exception(f'Output folder already exists: "{output_folder}"')
    log(f'Output folder: "{output_folder}"')

    sw_main = StopWatch('Main Experiments')

    experiments, settings = generate_experiment_list(experiments)
    save_setting(settings, Conf.FileLocations.SETTINGS_EXPERIMENT.substitute(name=f'{run_timestamp}'))

    exp_performances = ExperimentPerformances()
    for exp in experiments:
        try:
            exp_results = run_experiment(exp)
            exp_performances.add(exp_results)
        except Exception as e:
            log(f"Exception raised running experiment: {exp}", logging.ERROR)
            log_exception(e)

    sw_main.end()

    log('<<< FORMAT EXPLANATION OF PERFORMANCE VALUES >>> For each '
        'Experiment the results are shown for each iteration then the mean '
        'and standard deviation. For example [result_1, result_2, ..., '
        'result_n, mean, std_dev])')
    exp_performances.add_summary()

    # Save results to file
    perf_fn = Conf.FileLocations.PERFORMANCES.substitute(name=f'{run_timestamp}')
    exp_performances.save_to_file(perf_fn)
    log(f"Performances saved to: \"{perf_fn}\"")

    # Log Performance of Algorithms
    log(exp_performances)

    # Create table(s) with results
    if Conf.TempReports.SHOULD_GENERATE_RESULTS_AS_LATEX_TABLE:
        generate_results_latex_table(exp_performances, run_timestamp)
    if Conf.TempReports.SHOULD_GENERATE_RESULTS_AS_CSV:
        generate_results_csv(exp_performances, run_timestamp)

    # Log Runtime Settings
    log(f'Main Runtime was: {sw_main.runtime()}')

    log('\n<<<<<<<<<<<<<<<<<<<<<< COMPLETED >>>>>>>>>>>>>>>>>>>>>>')
    if Conf.MEMORY_TRACING:
        _current_size, peak_memory_usage = tracemalloc.get_traced_memory()
        log(f"Peak Memory usage was {peak_memory_usage / 1024 / 1024:.4f} MB")


if __name__ == '__main__':
    main_runner(main, should_complete_notify=Conf.SHOULD_COMPLETE_NOTIFY)
