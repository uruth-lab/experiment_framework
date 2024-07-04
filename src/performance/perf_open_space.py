import numpy as np
from opylib.log import log
from sklearn.ensemble import IsolationForest

from src.custom_algo.algorithms import CustomIF
from src.performance.perf_base import PerfBase, PerfConfig


class OpenSpaceDetector(PerfBase):
    """
    Checks if a brand exists along the axis aligned space outside the points. Returns a 1 for each threshold
    passed and a 0 if not passed
    """

    def _exec(self, exp_result):
        if exp_result.exp.algorithm.is_lof_without_novelty():
            trials = exp_result.trials_count
            return {
                f"Highest": [0] * trials,
                "f1": [0] * trials,
            }
        highest = []
        f1 = []

        for trial_idx, (clf, scores) in enumerate(exp_result.classifiers_and_scores()):
            trial_num = trial_idx + 1
            trial_highest = 0
            f1_highest = 0

            # Get F1 Threshold
            threshold_f1 = - self.get_best_threshold_or_default(exp_result.dataset.y, scores)

            # Add any additional points to the data and include their scores
            data = exp_result.dataset.X
            if isinstance(clf, CustomIF):
                additional_points = clf.additional_points
                additional_scores = exp_result.exp.algorithm.score(clf, additional_points)
                data = np.concatenate((data, clf.additional_points))
                scores = np.concatenate((scores, additional_scores))

            min_score = min(scores)
            max_score = max(scores)

            # Generate test points and score them
            offset = 1  # TODO Look into (and benchmark) if a larger value (dependent on the data) would produce
            #               faster test due to less failed extension
            feature_count = data.shape[1]
            for feat_ind in range(feature_count):
                if trial_highest >= 1:
                    # Already passed 100% we can stop looking now
                    break
                feat_values = [x[feat_ind] for x in data]
                min_ = min(feat_values)
                max_ = max(feat_values)
                for should_do_min in (True, False):
                    test_points = np.ndarray.copy(data)
                    for x in test_points:
                        # Set feature value specified by feat_ind to extreme value
                        x[feat_ind] = (min_ - offset) if should_do_min else (max_ + offset)

                    z = exp_result.exp.algorithm.score(clf, test_points)  # Score copied points
                    z = [(score, i) for i, score in enumerate(z)]  # Associate each score with the index of the point
                    z.sort()

                    # Find the point that has the highest score that extends
                    for score, point_index in reversed(z):
                        if self.check_band_extends(score, should_do_min, min_, max_, test_points[point_index], feat_ind,
                                                   exp_result.exp.algorithm.score, clf, trial_num):
                            trial_highest = max(trial_highest, (score - min_score) / (max_score - min_score))
                            if score >= threshold_f1:
                                f1_highest = max(f1_highest, 1)
                            break
            highest.append(trial_highest)
            f1.append(f1_highest)

        return {
            f"Highest": highest,
            "f1": f1,
        }

    @staticmethod
    def check_band_extends(threshold, should_do_min, min_, max_, test_point, feat_ind, score_fn, clf,
                           trial_num) -> bool:
        # WARNING: Before adapting this function for performance be very careful as semantics of extending any/all
        # points passed can get tricky and lead to errors. Last attempt resulted in an incorrect implementation that
        # checked for all points passing instead of any passing
        threshold_with_tolerance = threshold - PerfConfig.APX_EQUAL_THRESHOLD
        extend_base = max_ - min_
        assert extend_base > 0
        scalar_count = len(PerfConfig.OUTSIDE_OF_DATA_BAND_CONTINUOUS_CHECK_VALUES)
        extended_test_points = np.asarray([test_point] * scalar_count)
        for scalar_idx, scalar in enumerate(PerfConfig.OUTSIDE_OF_DATA_BAND_CONTINUOUS_CHECK_VALUES):
            offset = extend_base * scalar
            new_value = (min_ - offset) if should_do_min else (max_ + offset)
            extended_test_points[scalar_idx][feat_ind] = new_value
        extended_scores = score_fn(clf, extended_test_points)
        for i, extended_score in enumerate(extended_scores):
            if extended_score < threshold_with_tolerance:
                scalar = PerfConfig.OUTSIDE_OF_DATA_BAND_CONTINUOUS_CHECK_VALUES[i]
                log(f"Band Extension Test Failed on scalar {scalar} for threshold {threshold}")
                if isinstance(clf, IsolationForest):
                    point = extended_test_points[i]
                    extended_score = extended_scores[i]
                    raise AssertionError(
                        f"On trial ixd {trial_num} for isolation forest expected the same score for all points going "
                        f"out along the axis but found a discrepancy. When using the scalar {scalar}. Got point "
                        f"{point} which had a score of {extended_score} which is less than the threshold that the "
                        f"original point passed: {threshold}")
                return False
        return True
