from opylib.main_runner import main_runner

from src.config import Conf
from src.main_ import main

if __name__ == '__main__':
    main_runner(main, should_complete_notify=Conf.SHOULD_COMPLETE_NOTIFY)
