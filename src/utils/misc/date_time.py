from datetime import time


def get_run_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H%M")
