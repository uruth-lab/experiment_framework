from opylib.main_runner import main_runner

from src import main_, main_data_gen
from src.config import Conf
from src.supporting.no_depend import get_run_timestamp


def main():
    """
    Runs both the generation and the experiments
    """
    run_timestamp = get_run_timestamp()
    main_data_gen.main(run_timestamp)
    main_.main(run_timestamp)


if __name__ == "__main__":
    main_runner(main, should_complete_notify=Conf.SHOULD_COMPLETE_NOTIFY)
