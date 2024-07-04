import math
from string import Template

from src.enums import (TAlgorithmScoring, TAlgorithm, TDataset,
                       TDatasetGen,
                       TGroupDatasets, TGroupPerformance, TPerformance)


# ASSUMPTIONS:
# - GROUPS:
#   * [G01A01] Group label numbers are contiguous and start from 0

class Conf:
    SHOULD_COMPLETE_NOTIFY = True
    MEMORY_TRACING = False
    ANOM_LABEL = 1
    """ 
    Value  used in datasets when an instance is an anomaly
    (NB: I think this is assumed by some of the algorithms, only added for clarity of our code)
    """

    class TempReports:
        SHOULD_GENERATE_RESULTS_AS_LATEX_TABLE = True
        SHOULD_GENERATE_RESULTS_AS_CSV = True

    # TODO CONFIGURE DATASETS TO GENERATE HERE ############################
    """
    List of dataset composers to be called.
    Each dict overwrites default values if set.
    """
    DATASET_COMPOSERS = [  # Specific Composer - Highest precedence
        # {},  # Run with defaults
        {
            'output_filename': 'sine',
            'size': 500,
            'anom_ratio': 0,
            'gen_normal': [
                {'gen': TDatasetGen.GenSine,
                 'params': {
                     'should_add_noise': False,
                     # 'offsets': [1, 1],
                     # 'std_dev': [0.2, 0.2],
                 },
                 },
            ],
            'gen_anom': [
                {'gen': TDatasetGen.GenUniform,
                 'params': {},
                 },
            ],
        },
        # TODO 4 Generate old datasets
    ]

    # TODO CONFIGURE EXPERIMENTS HERE #########################################
    """
    List of experiments to be run.
    Each dict overwrites the default values if set
    """

    EXPERIMENTS = [  # Specific Experiment - Highest precedence
        # {},  # Run with defaults
        {
            'algorithms':
                [
                    {'algo': TAlgorithm.IsolationForest, 'params': {'random_state': 1, 'contamination': 1e-15}},
                    {'algo': TAlgorithm.CustomIF, 'params': {'random_state': 1, 'contamination': 1e-15}},
                    # {'algo': TAlgorithm.IsolationForest, 'params': {'contamination': 1e-15}},
                    # {'algo': TAlgorithm.CustomIF, 'params': {'contamination': 1e-15}},
                    {'algo': TAlgorithm.EIF, 'params': {}},
                    {'algo': TAlgorithm.PIDForest, 'params': {}},
                    {'algo': TAlgorithm.OneClassSVM, 'params': {}},
                    {'algo': TAlgorithm.LocalOutlierFactor, 'params': {}},
                    {'algo': TAlgorithm.LocalOutlierFactor, 'params': {"novelty": True}},
                ],
            'datasets': [
                TDataset.gaussian,
                TDataset.gaussian_exile,
                TDataset.gaussian_exile_poison,
                TDataset.gaussian_poison,
                TDataset.sine_uni,
                TDataset.sine,
                TDataset.sine_exile,
                TDataset.sine_exile_poison,
                TDataset.sine_poison,
                TDataset.diamond,
                TDataset.square,
                TDataset.tee_before,
                TDataset.tee_after,
                TDataset.normal_uni,
                TDataset.polynomial,
                TDataset.circle,
                TDataset.sphere,
                TDataset.single_dim,
                TDataset.anom_1feat,
                TDataset.http,
                TDataset.mammography,
                TDataset.musk,  # Has a lot of features (takes long for open space detection)
                TDataset.satimage2,  # Doesn't work for PID with open space
                TDataset.siesmic,
                TDataset.smtp,
                TDataset.thyroid,
                TDataset.vowels,  # Doesn't work for PID with open space
            ],
            'performance': [
                TPerformance.AvgPrecision,
                TPerformance.F1,
                TPerformance.FPR,
                # TPerformance.OpenSpaceDetector,
                TPerformance.PercentActualAnomaly,
                TPerformance.PercentPred,
                TPerformance.Precision,
                TPerformance.PrecisionRecallAUC,
                TPerformance.Recall_TPR,
                TPerformance.ROC_AUC,
                TPerformance.Scores,
                TPerformance.ScoresMin,
                TPerformance.ScoresMax,
                TPerformance.ThresholdF1,
                TPerformance.TimeExperiment,
                TPerformance.TimeTrial,
                TPerformance.VizPrecisionRecall,
                TPerformance.VizROC,
                TPerformance.VizScatter,
                TPerformance.VizScatterRaw,
                TPerformance.VizScores,
            ],
            'trials': 1,
            # Note if you specify one you have to specify all performance_config values (or changes to code are needed)
            # 'performance_config': {
            # }
        },
    ]

    class Projection:
        """
        When enabled all datasets will be projected to the specified dimension
        """
        ENABLED = False
        DIMENSION = 2
        # TODO: Add support for a random seed

    class Defaults:
        """
        Experiment Precedence High to Low
        - Specific Experiment
        - Default for Experiments
        - Default for algorithm

        Dataset Generation Precedence High to Low
        - Specific Composer
        - Defaults for Composer
        - Default for generator
        """

        EXPERIMENT = {  # Default for Experiments - Mid-Precedence
            # Takes a list of dicts of algorithms with their parameters
            # and a name for the algorithm with those settings (key: 'name')
            'algorithms': [
                {'algo': TAlgorithm.IsolationForest, 'params': {}},
                {'algo': TAlgorithm.CustomIF, 'params': {}},
                {'algo': TAlgorithm.LocalOutlierFactor, 'params': {}},
                {'algo': TAlgorithm.OneClassSVM, 'params': {}},
                {'algo': TAlgorithm.PIDForest, 'params': {}},
                {'algo': TAlgorithm.EIF, 'params': {}},
            ],

            # Takes a list of TDataset or a single TGroupDatasets
            'datasets': TGroupDatasets.all,

            # Takes a list of TPerformance or a single TGroupPerformance
            'performance': TGroupPerformance.all,

            # Number of times to run Nondeterministic algorithms
            'trials': 5,

            # Configurable Settings for Performance Measures
            'performance_config': {
                # Not currently used, was used for `open_space_intervals` but this setting was removed
            }
        }

        ALGORITHM_PARAMETERS = {  # Default for algorithms - Lowest Precedence
            TAlgorithm.IsolationForest: {},
            TAlgorithm.LocalOutlierFactor: {
                'n_neighbors': 20,
                'algorithm': 'auto',
                'leaf_size': 30,
                'metric': 'minkowski',
                'p': 2,
                'metric_params': None,
                'contamination': "auto",
                'novelty': False,
                'n_jobs': None,
            },
            TAlgorithm.OneClassSVM: {'kernel': 'rbf'},
            TAlgorithm.PIDForest: {'max_depth': 10,
                                   'n_trees': 50,
                                   'max_samples': 100,
                                   'max_buckets': 3,
                                   'epsilon': 0.1,
                                   'sample_axis': 1,
                                   'threshold': 0},
            TAlgorithm.EIF: {
                # Sample notebook https://github.com/sahandha/eif/blob/master/Notebooks/EIF.ipynb
                'ntrees': 100,
                'sample_size': 256,
                'ExtensionLevel': None,  # None means dim data less 1 (This is custom and implemented in our code)
            },
            TAlgorithm.Custom1: {},
            TAlgorithm.CustomIF: {},  # See constructor for CustomIF for options available
        }

        """"
        Default setting for dataset composer. Composer takes a 
        composition
        of generators and joins them together to make one dataset.
        Values set here take precedence over individual 
        generator values.
        """
        DATASET_COMPOSER = {  # Defaults for Composer - Mid-precedence
            # This size is the total dataset size, unless in the params for
            # a dataset (one in gen_normal or gen_anom) has a value set it
            # will not be replaced. Otherwise, the normal portion will be
            # split evenly among gen_normal and anom portion evenly among
            # gen_anom
            'size': 100,

            'anom_ratio': .1,  # Percentage of points that are anom
            'should_shuffle': True,
            'output_filename': None,  # WARNING: Will overwrite existing files

            # Provide easy way to set consistent number of features
            'feats_informative': 2,
            'feats_irrelevant': 0,

            # Takes a list of dicts of generators with their parameters and
            # optional post generation transformation
            'gen_normal': [
                {'gen': TDatasetGen.GenGauss, 'params': {}, 'transform': None}
            ],

            # Takes a list of dicts of generators with their parameters and
            # optional post generation transformation
            #
            # Also optionally supports another parameter specifying another
            # generator to feed points to the outer generator.
            'gen_anom': [
                {'gen': TDatasetGen.GenUniform, 'params': {},
                 'transform': None, 'feeder': None}
            ],
        }

        DATASET_GEN_SETTING = {  # Default for generator  - Lowest precedence
            # Parent class values have lower precedence
            TDatasetGen.GenBase: {
                # 'size': 25,
                'group': 0,
                'feats_informative': 2,
                'feats_irrelevant': 0,
                'irrelevant_feat_range': [-1, 1],
                'should_add_noise': True,
                'noise_sd': [0.5],
                'max_abs_noise': -1,  # Ignored if negative
                'rej_sd_coefficient': 2,
                'offsets': [0],  # Last value repeated as needed

                # Max allowed ratio of rejected points to accepted points
                'anom_max_rej_ratio': 10,
            },
            TDatasetGen.GenBaseFunc: {
                'x_min_max': [0, 10],
                'invert_xy': False,  # Treat X as Y
            },
            TDatasetGen.GenUniform: {
                'feat_range': [[0, 10], ],  # Last value repeated as needed
            },
            TDatasetGen.GenPolynomial: {
                # High order first. Eg [8,-2,5] is 8x^2-2x+5
                'coefficients': [1, 0, ],
            },
            TDatasetGen.GenSine: {
                'x_min_max': [0, 4 * math.pi],
            },
            TDatasetGen.GenSphere: {
                'noise_sd': [.01],  # Last value repeated as needed
                'radius': 2,
                'fixed_angles': {},  # Key is dimen, value is fixed angle
            },
            TDatasetGen.GenGauss: {
                'std_dev': [1],  # Last value repeated as needed
                'should_add_noise': False,
            },
        }

    class Groups:
        """
        Provides mappings from groups enums to lists of values
        """

        class Performance:  # Mapping for TGroupPerformance
            numeric_only = [
                TPerformance.AvgPrecision,
                TPerformance.F1,
                TPerformance.FPR,
                TPerformance.OpenSpaceDetector,
                TPerformance.PercentActualAnomaly,
                TPerformance.PercentPred,
                TPerformance.Precision,
                TPerformance.PrecisionRecallAUC,
                TPerformance.Recall_TPR,
                TPerformance.ROC_AUC,
                TPerformance.Scores,
                TPerformance.ScoresMin,
                TPerformance.ScoresMax,
                TPerformance.ThresholdF1,
                TPerformance.TimeExperiment,
                TPerformance.TimeTrial,
                TPerformance.EqualizedOdds,
            ]

            all = [x for x in TPerformance]

        class Datasets:  # Mapping for TGroupDatasets
            synthetic = [
                TDataset.gaussian,
                TDataset.gaussian_exile,
                TDataset.gaussian_exile_poison,
                TDataset.gaussian_poison,
                TDataset.sine_uni,
                TDataset.sine,
                TDataset.sine_exile,
                TDataset.sine_exile_poison,
                TDataset.sine_poison,
                TDataset.diamond,
                TDataset.square,
                TDataset.tee_before,
                TDataset.tee_after,
                TDataset.normal_uni,
                TDataset.single_dim,
                TDataset.anom_1feat,
                TDataset.polynomial,
                TDataset.circle,
                TDataset.sphere,
            ]

            real = [
                TDataset.http,
                TDataset.mammography,
                TDataset.musk,
                TDataset.satimage2,
                TDataset.siesmic,
                TDataset.smtp,
                TDataset.thyroid,
                TDataset.vowels,
            ]

            all = [x for x in TDataset]

    class Algorithms:
        """
        Stores invariant properties of the algorithms
        Steps to add algorithm:
            1 - Add algorithm to TAlgorithm enum (specifies class)
            2 - Add algorithm scoring function to TAlgorithmScoring
            3 - Add mapping for default function parameters in
                Conf.Defaults.ALGORITHM_PARAMETERS
            4 - Add mapping for Scoring function in
                Conf.Algorithms.SCORING
            5 - If the function is deterministic add to
                Conf.Algorithms.DETERMINISTIC
            6 - If the function needs the data to be transposed then add to
                Conf.Algorithms.REQ_TRANSPOSE
            7 - Add to default set of algorithms in
                Conf.Defaults.EXPERIMENT['algorithms']
        """

        "Stores mapping of algorithm to scoring method"
        SCORING = {
            TAlgorithm.IsolationForest: TAlgorithmScoring.IsolationForest,
            TAlgorithm.LocalOutlierFactor:
                TAlgorithmScoring.LocalOutlierFactor,
            TAlgorithm.OneClassSVM: TAlgorithmScoring.OneClassSVM,
            TAlgorithm.PIDForest: TAlgorithmScoring.PIDForest,
            TAlgorithm.EIF: TAlgorithmScoring.EIF,
            TAlgorithm.Custom1: TAlgorithmScoring.Custom1,
            TAlgorithm.CustomIF: TAlgorithmScoring.IsolationForest,
        }

        "Items in this set are deterministic"
        DETERMINISTIC = {
            TAlgorithm.LocalOutlierFactor,
            TAlgorithm.OneClassSVM,
        }

        "Items in this set require the data to be transposed"
        REQ_TRANSPOSE = {
            TAlgorithm.PIDForest,
        }

    class FileLocations:
        RESULTS = 'results/'
        DATASETS = Template('data/${name}.mat')
        PERFORMANCES = Template(RESULTS + '${name}_performances.yaml')
        RESULTS_CSV = Template(RESULTS + '${name}_results.csv')
        RESULTS_TABLE_LATEX = Template(RESULTS + '${name}_latex_results_table.txt')
        SETTINGS_EXPERIMENT = Template(
            RESULTS + '${name}_ExperimentSettings.yaml')
        SETTINGS_COMPOSER = Template(RESULTS + '${name}_ComposerSettings.yaml')
