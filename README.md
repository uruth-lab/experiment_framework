# An Axiomatic Perspective on Anomaly Detection

This repository host the experimental framework used in the development of the Thesis of Chester Wyke.

## Credits

- Includes code for PIDForest from this [paper](https://arxiv.org/abs/1912.03582). Code copied from
  their [GitHub](https://github.com/vatsalsharan/pidforest) repo.
- Real world UCI datasets were copied from [PIDForest Github](https://github.com/vatsalsharan/pidforest)

## How to use

### Setup

- Clone / Download the Code
- [Optional] Set up a virtual environment. I used [venv](https://docs.python.org/3/library/venv.html), but you can
  choose
  any that you're familiar with. The instructions below use pip because of this but just substitute as applicable.
- Install dependencies
    - Dependencies are listed in [requirements.txt](requirements.txt)
    - You can install using `pip install -r requirements.txt` from the root of the repository folder.

### Running experiments

- Configure the experiments desired in [src/config.py](src/config.py). See section after the line that
  says `EXPERIMENTS = [  # Specific Experiment - Highest precedence`
- Then run `run_experiment.py` from the root of the repository. The command should be `python run_experiment.py`
- The results will be placed in [results](results) folder and the logs will go to [log](log).
  If there are any errors they will go to a separate file in the logs folder called`ERRORS.log`

### Generating Datasets

- Configure the dataset to be generated in [src/config.py](src/config.py). See section after the line that
  says `DATASET_COMPOSERS = [  # Specific Composer - Highest precedence`
- Then run `run_gerate.py` from the root of the repository. The command should be `python run_generate.py`
- The generated dataset will be placed in the [data](data) folder with the name you specified with '.mat' appended (It
  will be a MatLab file).