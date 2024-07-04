import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.performance.perf_base import PerfConfig, PerfGraphic


class VizScores(PerfGraphic):
    def _exec(self, exp_result):
        count_points, count_feat = exp_result.dataset.X.shape

        y = exp_result.dataset.y

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

        # Build array of point sizes to use for masking which point to plot
        point_size = matplotlib.rcParams['lines.markersize'] ** 2
        sizes = np.empty(count_points)
        sizes.fill(point_size)

        # Get mask for ground truth anomalies
        indices_anom_ground = np.where(y == 1)[0]
        mask_anom_ground = np.zeros(count_points, bool)
        mask_anom_ground[indices_anom_ground] = True

        all_zeros = np.zeros(count_points)

        for i, scores in enumerate(exp_result.trials_scores):
            # Get performance metrics for title
            f1, max_ind, precisions, recalls, thresholds = \
                self.get_best_threshold_full(exp_result.dataset.y, scores)
            if max_ind < 0:
                # Set to value above max (effectively disable it)
                threshold = max(scores) + 1
                perf_metrics = ''
            else:
                threshold = thresholds[max_ind]
                precision = precisions[max_ind]
                rec = recalls[max_ind]
                perf_metrics = '\nF1: %.3f | Precision: %.3f | Recall: %.3f' \
                               % (
                                   f1, precision, rec)

            # Get mask for predicated anomalies
            indices_anom_pred = np.where(-scores >= threshold)
            mask_anom_pred = np.zeros(count_points, bool)
            mask_anom_pred[indices_anom_pred] = True

            # Plot True Positive (TP)
            mask_to_plot = mask_anom_ground & mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.figure(dpi=PerfConfig.VIZ_DPI)
            plt.scatter(-scores, all_zeros, label=tp_label, s=s,
                        marker=PerfConfig.ScatterPlot.MARKER_TP,
                        c=PerfConfig.ScatterPlot.COLOR_TP)

            # Plot False Positive (FP)
            mask_to_plot = ~mask_anom_ground & mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.scatter(-scores, all_zeros, label=fp_label, s=s,
                        marker=PerfConfig.ScatterPlot.MARKER_FP,
                        c=PerfConfig.ScatterPlot.COLOR_FP)

            # Plot True Negative (TN)
            mask_to_plot = ~mask_anom_ground & ~mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.scatter(-scores, all_zeros, label=tn_label, s=s,
                        marker=PerfConfig.ScatterPlot.MARKER_TN,
                        c=PerfConfig.ScatterPlot.COLOR_TN)

            # Plot False Negative (FN)
            mask_to_plot = mask_anom_ground & ~mask_anom_pred
            s = np.ma.masked_array(sizes, mask=~mask_to_plot)
            plt.scatter(-scores, all_zeros, label=fn_label, s=s,
                        marker=PerfConfig.ScatterPlot.MARKER_FN,
                        c=PerfConfig.ScatterPlot.COLOR_FN)

            # Plot threshold
            plt.axvline(x=threshold)

            if PerfConfig.SHOW_HEADINGS:
                plt.title(f'{exp_result} Scores Plot' + perf_metrics)
            plt.legend()
            if PerfConfig.VIZ_SAVE_TO_DISK:
                plt.savefig(self.viz_file_name(exp_result.exp, i + 1))
            if PerfConfig.VIZ_DISPLAY:
                plt.show()
            plt.close()
