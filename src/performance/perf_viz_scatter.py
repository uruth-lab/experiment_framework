from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.custom_algo.algorithms import CustomIF
from src.performance.perf_base import PerfConfig
from src.performance.perf_viz_scatter_raw import VizScatterRaw


class Cache:
    def __init__(self):
        self.extent = None
        self.points_to_score_for_heatmap = None
        self.heatmap_shape = None


# noinspection PyPep8Naming
class VizScatter(VizScatterRaw):
    def _exec(self, exp_result):
        if PerfConfig.RELEASE_MODE:
            shared_color = '#FF7700'
            shared_marker = '.'
            color_tp = '#FF0000'
            color_fp = shared_color
            color_tn = shared_color
            color_fn = color_tp
            color_extra = shared_color
            marker_tp = '*'
            marker_fp = shared_marker
            marker_tn = shared_marker
            marker_fn = marker_tp
            marker_extra = shared_marker
            heatmap_color = 'GnBu'
        else:
            color_tp = PerfConfig.ScatterPlot.COLOR_TP
            color_fp = PerfConfig.ScatterPlot.COLOR_FP
            color_tn = PerfConfig.ScatterPlot.COLOR_TN
            color_fn = PerfConfig.ScatterPlot.COLOR_FN
            color_extra = PerfConfig.ScatterPlot.COLOR_EXTRA
            marker_tp = PerfConfig.ScatterPlot.MARKER_TP
            marker_fp = PerfConfig.ScatterPlot.MARKER_FP
            marker_tn = PerfConfig.ScatterPlot.MARKER_TN
            marker_fn = PerfConfig.ScatterPlot.MARKER_FN
            marker_extra = PerfConfig.ScatterPlot.MARKER_EXTRA
            heatmap_color = PerfConfig.ScatterPlot.HEATMAP_COLORMAP
        if PerfConfig.LONG_LABELS:
            tp_label = "True Positive"
            fp_label = "False Positive"
            tn_label = "True Negative"
            fn_label = "False Negative"
        else:
            tp_label = "TP"
            fp_label = "FP"
            tn_label = "TN"
            fn_label = "FN"
        cache: Optional[Cache] = None
        count_points, count_feat = exp_result.dataset.X.shape

        exp = exp_result.exp
        should_do_heatmap = (
                PerfConfig.ScatterPlot.HEATMAP_ENABLED and
                count_feat <= 2 and not exp.algorithm.is_lof_without_novelty()
        )

        y = exp_result.dataset.y
        X = exp_result.dataset.X
        feat2 = self.feat2_determine(count_feat, count_points, X)

        # Build array of point sizes to use for masking which point to plot
        point_size = self.point_size
        sizes = np.empty(count_points)
        sizes.fill(point_size)

        # Get mask for ground truth anomalies
        indices_anom_ground = np.where(y == 1)[0]
        mask_anom_ground = np.zeros(count_points, bool)
        mask_anom_ground[indices_anom_ground] = True

        for i, scores in enumerate(exp_result.trials_scores):
            if PerfConfig.ScatterPlot.ARTIFICIALLY_BOOST_ABOVE_THRESHOLD:
                separation_value = round(max(scores) - min(scores), 3)  # Arbitrarily picked value
                separation_info = f' (Arbitrary Separation of {separation_value} added)'
            else:
                separation_info = ''
                separation_value = 0

            # Get performance metrics for title
            f1, max_ind, precisions, recalls, thresholds = \
                self.get_best_threshold_full(exp_result.dataset.y, scores)
            if max_ind < 0:
                threshold = - self.get_default_threshold(scores)
                perf_metrics = ''
                # TODO 4: Add some kind of warning here to be able to know this code ran
            else:
                threshold = -thresholds[max_ind]
                precision = precisions[max_ind]
                rec = recalls[max_ind]
                perf_metrics = '\nF1: %.3f | Precision: %.3f | Recall: %.3f' % (f1, precision, rec)

            # Get mask for predicated anomalies
            indices_anom_pred = np.where(scores <= threshold)
            mask_anom_pred = np.zeros(count_points, bool)
            mask_anom_pred[indices_anom_pred] = True

            # Plot True Positive (TP)
            mask_to_plot = mask_anom_ground & mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            fig = plt.figure(dpi=PerfConfig.VIZ_DPI)
            plt.scatter(X[:, 0], feat2(), label=tp_label, s=s, marker=marker_tp, c=color_tp)

            # Plot False Positive (FP)
            mask_to_plot = ~mask_anom_ground & mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.scatter(X[:, 0], feat2(), label=fp_label, s=s, marker=marker_fp, c=color_fp)

            # Plot True Negative (TN)
            mask_to_plot = ~mask_anom_ground & ~mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.scatter(X[:, 0], feat2(), label=tn_label, s=s, marker=marker_tn, c=color_tn)

            # Plot False Negative (FN)
            mask_to_plot = mask_anom_ground & ~mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.scatter(X[:, 0], feat2(), label=fn_label, s=s, marker=marker_fn, c=color_fn)

            if isinstance(exp.classifiers[i], CustomIF):
                extra_points = exp.classifiers[i].additional_points
                # TODO 1: Add option to disable these points showing
                # TODO 1: Generate point sizes list more efficiently
                # TODO 1: Change size of extra points?
                plt.scatter(extra_points[:, 0], self.feat2_determine(count_feat, len(extra_points), extra_points)(),
                            label='Extra', s=[point_size] * len(extra_points), marker=marker_extra,
                            c=color_extra)

            if PerfConfig.SHOW_HEADINGS:
                plt.title(f'{exp_result} Scatter Plot' +
                          ('' if count_feat <= 2 else f', 2 of {count_feat} feat')
                          + perf_metrics)

            if not PerfConfig.RELEASE_MODE:
                plt.grid(True)
                plt.legend()
            x_limits = plt.xlim()
            y_limits = plt.ylim()

            if PerfConfig.ScatterPlot.SAME_SCALE:
                both_limits = x_limits + y_limits
                new_limits = [min(both_limits), max(both_limits)]
                x_limits = plt.xlim(new_limits[0], new_limits[1])
                y_limits = plt.ylim(new_limits[0], new_limits[1])

            if PerfConfig.ScatterPlot.EQUAL_ASPECT:
                fig.axes[0].set_aspect('equal', adjustable='box')

            # Resize plot
            if PerfConfig.ScatterPlot.OPEN_SPACE_PERCENTAGE > 0:
                increase_percentage = PerfConfig.ScatterPlot.OPEN_SPACE_PERCENTAGE / 100
                x_size = x_limits[1] - x_limits[0]
                space_size = x_size * increase_percentage
                x_limits = plt.xlim(x_limits[0] - space_size, x_limits[1] + space_size)
                y_size = y_limits[1] - y_limits[0]
                space_size = y_size * increase_percentage
                y_limits = plt.ylim(y_limits[0] - space_size, y_limits[1] + space_size)

            # Heat Map
            if should_do_heatmap:
                clf = exp.classifiers[i]
                if cache is None:
                    cache = Cache()
                    cache.extent, cache.points_to_score_for_heatmap, cache.heatmap_shape = self.get_heatmap_points(
                        count_feat,
                        exp.algorithm.needs_transposed,
                        x_limits,
                        y_limits)
                extent, points_to_score_for_heatmap, heatmap_shape = (
                    cache.extent, cache.points_to_score_for_heatmap, cache.heatmap_shape
                )

                # Calculate scores for heatmap
                z = self.score_points_for_heatmap(heatmap_shape, points_to_score_for_heatmap, clf, exp.algorithm.score)

                # Calculate ticks for color bar
                heatmap_min = np.min(z)
                heatmap_lower = round(heatmap_min, 3)
                if heatmap_lower < heatmap_min:
                    heatmap_lower += 0.001
                heatmap_max = np.max(z)
                heatmap_upper = round(heatmap_max, 3)
                if heatmap_upper > heatmap_max:
                    heatmap_upper -= 0.001
                scores_lower = round(np.min(scores), 3)
                is_scores_below_heatmap = scores_lower < heatmap_min
                scores_upper = round(np.max(scores), 3)
                is_scores_above_heatmap = scores_upper > heatmap_max
                ticks = np.linspace(scores_lower, scores_upper, PerfConfig.ScatterPlot.COLOR_BAR_TICK_COUNT)
                ticks = [round(x, 3) for x in ticks] + [round(threshold, 3)]
                if PerfConfig.ScatterPlot.INCLUDE_HEATMAP_BOUNDARY_VALUES_AS_TICKS:
                    heatmap_endpoints = [heatmap_lower, heatmap_upper]
                    ticks = ticks + heatmap_endpoints

                if PerfConfig.ScatterPlot.FLOOR_SCORES_FOR_COLORS_ONLY_TO_TICKS:
                    ticks.sort()  # Code below assumes sorted list (the threshold added may not be in the right place)
                    for row in range(len(z)):
                        for col in range(len(z[row])):
                            for tick_index in range(1, len(ticks)):
                                if ticks[tick_index] > z[row][col]:
                                    z[row][col] = ticks[tick_index - 1]
                                    break
                            else:
                                z[row][col] = ticks[-1]

                    if PerfConfig.ScatterPlot.ARTIFICIALLY_BOOST_ABOVE_THRESHOLD:
                        rounded_threshold = round(threshold, 3)
                        for tick_index in range(len(ticks)):
                            if ticks[tick_index] < rounded_threshold:
                                # Move ticks less than threshold down.
                                # Leaving threshold to make it easy to find even though points on the
                                # boundary are treated as anomalies
                                ticks[tick_index] -= separation_value
                        for row_ in range(len(z)):
                            for col_ in range(len(z[row_])):
                                if z[row_][col_] <= rounded_threshold:
                                    # Move points below threshold down to make the separation easier to see
                                    # (including points on the boundary so that when all points are treated as anomalies
                                    # because there are no positive "normal" points  then the colors don't look weird)
                                    z[row_][col_] -= separation_value
                        if scores_lower < rounded_threshold:
                            scores_lower -= separation_value
                        if scores_upper < rounded_threshold:
                            scores_upper -= separation_value

                # Show min/max below x-axis
                if not PerfConfig.RELEASE_MODE:
                    superscript_one = "\u00b9"
                    lower_off_chart = superscript_one if is_scores_below_heatmap else ""
                    upper_off_chart = superscript_one if is_scores_above_heatmap else ""
                    explanation = f" ({superscript_one} means off scale)" \
                        if is_scores_above_heatmap or is_scores_below_heatmap else ""
                    lower = f"{scores_lower:.3f}{lower_off_chart}"
                    upper = f"{scores_upper:.3f}{upper_off_chart}"
                    plt.xlabel(f"Training Score Min: {lower} Max: {upper}{explanation}")

                # Plot heatmap
                im = plt.imshow(z, cmap=plt.cm.get_cmap(heatmap_color), interpolation='None', extent=extent)

                # Create color bar
                if not PerfConfig.RELEASE_MODE:
                    cbar = plt.colorbar(im, ticks=ticks)
                    cbar.ax.set_ylabel(f"Score (thresh: {threshold:.3f}){separation_info}", rotation=-90, va="bottom")

            if PerfConfig.VIZ_SAVE_TO_DISK:
                plt.savefig(self.viz_file_name(exp_result.exp, i + 1))
            if PerfConfig.VIZ_DISPLAY:
                plt.show()
            plt.close()

    @staticmethod
    def score_points_for_heatmap(heatmap_shape, points_to_score_for_heatmap, clf, score_fn):
        result = score_fn(clf, points_to_score_for_heatmap)
        result = result.reshape(heatmap_shape)
        return result

    @staticmethod
    def get_heatmap_points(count_feat, needs_transposed, x_limits, y_limits):
        xx0, xx1 = np.meshgrid(
            np.linspace(x_limits[0], x_limits[1], PerfConfig.ScatterPlot.HEATMAP_POINTS_PER_AXIS),
            np.flip(np.linspace(y_limits[0], y_limits[1], PerfConfig.ScatterPlot.HEATMAP_POINTS_PER_AXIS)))
        heatmap_shape = xx0.shape
        if count_feat == 2:
            points_to_score_for_heatmap = np.c_[xx0.ravel(), xx1.ravel()]
            extent = np.min(xx0), np.max(xx0), np.min(xx1), np.max(xx1)
        elif count_feat == 1:
            points_to_score_for_heatmap = np.c_[xx0.ravel()]
            extent = np.min(xx0), np.max(xx0), np.min(xx0), np.max(xx0)
        else:
            # Should only be 1 or 2 based on check at top that enables heatmap
            assert False, f"Expected number of features to be 1 or 2 but got {count_feat}"
        if needs_transposed:
            points_to_score_for_heatmap = points_to_score_for_heatmap.transpose()
        return extent, points_to_score_for_heatmap, heatmap_shape
