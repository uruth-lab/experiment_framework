from opylib.log import log
from opylib.main_runner import main_runner
from opylib.stopwatch import StopWatch

from src.config import Conf
from src.data_generator.dataset_composer import DatasetComposer
from src.supporting.no_depend import get_run_timestamp
from src.utils.misc.data_import_export import save_yaml


def main(run_timestamp=None):
    """
    Runs generation of the configured datasets (see config.py)
    """
    if run_timestamp is None:
        run_timestamp = get_run_timestamp()
    log(f"Called from main_data_gen.py {run_timestamp=} (To make it easier to associate with other output)")
    all_settings = []  # To store composer setting for saving to disk
    sw_main = StopWatch('Main Data Generation')
    for i, composer_settings in enumerate(Conf.DATASET_COMPOSERS, 1):
        composer = DatasetComposer(i, composer_settings)
        all_settings.append({
            'Name': f'{composer}',
            'Value': {**composer.settings, 'ID': i}
        })
        sw_compose = StopWatch(f'Start {composer}')
        composer.generate()
        sw_compose.end()
        log('')  # Create spacing for readability
    sw_main.end()
    save_yaml(all_settings,
              Conf.FileLocations.SETTINGS_COMPOSER.substitute(
                  name=f'{run_timestamp}'))
    log('\n<<<<<<<<<<<<<<<<<<<<<< COMPLETED >>>>>>>>>>>>>>>>>>>>>>')


if __name__ == "__main__":
    main_runner(main, should_complete_notify=Conf.SHOULD_COMPLETE_NOTIFY)
