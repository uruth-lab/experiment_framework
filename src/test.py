from opylib.log import setup_log
from opylib.stopwatch import StopWatch


def main():
    setup_log(only_std_out=True)
    sw = StopWatch("Test")
    endpoint = 10
    for i in range(endpoint):
        print(f'i={i} endpoint = {endpoint}')
        endpoint -= 1
    sw.end()


if __name__ == "__main__":
    main()
