import enum
from enum import Enum, auto


###############################################################################
# GROUPS (Expected value is attribute of Conf)
@enum.unique
class TGroupPerformance(Enum):
    numeric_only = 'Groups.Performance.numeric_only'
    all = 'Groups.Performance.all'


@enum.unique
class TGroupDatasets(Enum):
    synthetic = 'Groups.Datasets.synthetic'
    real = 'Groups.Datasets.real'
    all = 'Groups.Datasets.all'


###############################################################################
# DATASETS
@enum.unique
class TDataset(Enum):
    gaussian = 'gaussian'
    gaussian_exile = 'gaussian_exile'
    gaussian_exile_poison = 'gaussian_exile_poison'
    gaussian_poison = 'gaussian_poison'
    sine_uni = 'sine_uni'  # Represents our original sine dataset before sine was repurposed for LOF experiment
    sine = 'sine'
    sine_exile = 'sine_exile'
    sine_exile_poison = 'sine_exile_poison'
    sine_poison = 'sine_poison'
    diamond = 'diamond'
    square = 'square'
    tee_before = 'tee_before'
    tee_after = 'tee_after'
    normal_uni = 'normal_uni'
    single_dim = 'single_dim'
    anom_1feat = 'anom_1feat'  # Called "all-but-one" in thesis
    polynomial = 'polynomial'
    circle = 'circle'
    sphere = 'sphere'
    # From PID Paper (Also found on http://odds.cs.stonybrook.edu/)
    http = 'http'
    mammography = 'mammography'
    musk = 'musk'
    satimage2 = 'satimage-2'  # Doesn't work for PID with open space
    siesmic = 'siesmic'  # Doesn't work with EIF because EIF doesn't support int
    smtp = 'smtp'
    thyroid = 'thyroid'
    vowels = 'vowels'  # Doesn't work for PID with open space


###############################################################################
# DATASET GENERATION
@enum.unique
class TDatasetGen(Enum):
    # ABSTRACT CLASSES (Has Base in Name)
    GenBase = 'src.data_generator.gen_base.GenBase'
    GenBaseFunc = 'src.data_generator.gen_base_func.GenBaseFunc'

    # CONCRETE CLASSES
    GenGauss = 'src.data_generator.gen_gauss.GenGauss'
    GenPolynomial = 'src.data_generator.gen_polynomial.GenPolynomial'
    GenSine = 'src.data_generator.gen_sine.GenSine'
    GenSphere = 'src.data_generator.gen_sphere.GenSphere'
    GenUniform = 'src.data_generator.gen_uniform.GenUniform'
    # TODO:4 Add type that adds noise to normal points


###############################################################################
# ALGORITHMS
@enum.unique
class TAlgorithm(Enum):
    IsolationForest = 'sklearn.ensemble.IsolationForest'
    LocalOutlierFactor = 'sklearn.neighbors.LocalOutlierFactor'
    OneClassSVM = 'sklearn.svm.OneClassSVM'
    PIDForest = 'src.pid_forest.pid_forest.PIDForest'
    EIF = 'eif.iForest'
    Custom1 = 'src.custom_algo.algorithms.CustomAlgo1'
    CustomIF = 'src.custom_algo.algorithms.CustomIF'

    @staticmethod
    def from_name(name: str):
        for x in TAlgorithm:
            if x.name == name:
                return x
        raise Exception(f"Unknown algorithm name: {name}")


@enum.unique
class TAlgorithmScoring(Enum):
    IsolationForest = 'src.supporting.scoring_methods.score_iso'
    LocalOutlierFactor = 'src.supporting.scoring_methods.score_lof'
    OneClassSVM = 'src.supporting.scoring_methods.score_oc_svm'
    PIDForest = 'src.supporting.scoring_methods.score_pid'
    EIF = 'src.supporting.scoring_methods.score_eif'
    Custom1 = 'src.supporting.scoring_methods.score_custom1'


@enum.unique
class TPerformance(Enum):
    AvgPrecision = 'src.performance.perf_avg_precision.AvgPrecision'
    F1 = 'src.performance.perf_f1.F1'
    FPR = 'src.performance.perf_fpr.FPR'
    OpenSpaceDetector = 'src.performance.perf_open_space.OpenSpaceDetector'
    "Designed to search for brands produced by isolation fores"
    PercentActualAnomaly = \
        'src.performance.perf_per_actual_anomaly.PercentActualAnomaly'
    PercentPred = 'src.performance.perf_per_pred.PercentPred'
    Precision = 'src.performance.perf_precision.Precision'
    PrecisionRecallAUC = 'src.performance.perf_precision_recall_auc' \
                         '.PrecisionRecallAUC'
    Recall_TPR = 'src.performance.perf_recall.Recall_TPR'
    ROC_AUC = 'src.performance.perf_roc_auc.ROC_AUC'
    Scores = 'src.performance.perf_scores.Scores'
    ScoresMin = 'src.performance.perf_scores.ScoresMin'
    ScoresMax = 'src.performance.perf_scores.ScoresMax'
    ThresholdF1 = 'src.performance.perf_threshold.ThresholdF1'
    TimeExperiment = 'src.performance.perf_time_exp.TimeExperiment'
    TimeTrial = 'src.performance.perf_time_trial.TimeTrial'
    EqualizedOdds = 'src.performance.perf_equ_odds.EqualizedOdds'
    VizPrecisionRecall = 'src.performance.perf_viz_precision_recall' \
                         '.VizPrecisionRecall'
    VizROC = 'src.performance.perf_viz_roc.VizROC'
    VizScatter = 'src.performance.perf_viz_scatter.VizScatter'
    VizScatterRaw = 'src.performance.perf_viz_scatter_raw.VizScatterRaw'
    VizScores = 'src.performance.perf_viz_scores.VizScores'


###############################################################################
# MISC
@enum.unique
class TBeepMode(Enum):
    DISABLED = auto()
    WINDOWS = auto()
    COLAB = auto()


@enum.unique
class TCustomIFMode(Enum):
    SINGLE_SPECIAL = auto()
    SINGLE_RANDOM = auto()
    SQUARE_GRID = auto()
    MULTIPLE_RANDOM = auto()
    LINE_IN_MIDDLE = auto()
    CIRCLE_IN_MIDDLE = auto()
    DISK_IN_MIDDLE = auto()
    MIN_AND_MAX = auto()
    ALL_MAX = auto()


@enum.unique
class TCircleRadiusReference(Enum):
    MIN_DIMENSION = auto()
    AVERAGE_DIMENSION = auto()
    MAX_DIMENSION = auto()
