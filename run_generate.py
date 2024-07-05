from opylib.main_runner import main_runner

from src.config import Conf
from src.main_data_gen import main

if __name__ == "__main__":
    main_runner(main, should_complete_notify=Conf.SHOULD_COMPLETE_NOTIFY)
