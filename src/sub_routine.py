"""
Entry point for integration with external drivers like CLI or rust code
"""
import argparse
import logging
from typing import List, Tuple, Dict

from opylib.log import log
from opylib.main_runner import main_runner

from src import main_
from src.enums import TAlgorithm, TPerformance, TDataset


class Defaults:
    """
    Default values for sub routine runs
    """
    ALGORITHMS = [(TAlgorithm.IsolationForest, {})]
    PERFORMANCE = [TPerformance.VizScatter, TPerformance.Scores, ]


def run_experiment(datasets: List[TDataset | str], algorithms: List[Tuple[TAlgorithm, Dict]] = None,
                   performance: List[TPerformance] = None, trials: int = 1):
    """
    Calls out to the regular main passing in experiments instead of using the value from Conf.
    Defaults from Conf are still used where not overridden
    :return:
    """
    if len(datasets) == 0:
        raise Exception("No datasets found")
    if algorithms is None:
        algorithms = Defaults.ALGORITHMS
    if performance is None:
        performance = Defaults.PERFORMANCE
    experiment = {
        'algorithms': [{'algo': x[0], 'params': x[1]} for x in algorithms],
        'datasets': datasets,
        'performance': performance,
        'trials': trials,
    }
    main_.main(experiments=[experiment])


def main():
    """
    Handles runs from the command line
    https://docs.python.org/3/library/argparse.html
    :return:
    """

    # Handle CLI arguments
    parser = argparse.ArgumentParser(
        prog='PYTHONPATH=$(pwd) python src/sub_routine.py',
        description='CLI Interface to run experiments',
        epilog='Should be run from folder containing "src". '
               'Note a cross product of algorithms and datasets is performed. (All datasets for all algorithms)'
    )
    parser.add_argument("datasets", help="list of one or more datasets to run, separated by commas")
    parser.add_argument('-a', '--algorithms', help="list of one or more algorithms to use, separated by commas")
    args = parser.parse_args()

    # Log raw values received
    log(f"Raw datasets value {args.datasets}", log_level=logging.DEBUG)
    log(f"Raw algorithms value {args.algorithms}", log_level=logging.DEBUG)

    # Remove .mat if included in the input
    datasets = [x if not x.endswith(".mat") else ".".join(x.split(".")[0:-1]) for x in args.datasets.split(",")]
    log(f"Datasets from command line: {datasets}")

    # Convert algorithms into the corresponding enum types
    algorithms = None if args.algorithms is None else [(TAlgorithm.from_name(x), {}) for x in
                                                       args.algorithms.split(",")]
    log(f"Algorithms from command line: {algorithms}")

    # TODO 1: Ensure datasets exist before calling run

    run_experiment(datasets, algorithms=algorithms)


if __name__ == '__main__':
    main_runner(main)
