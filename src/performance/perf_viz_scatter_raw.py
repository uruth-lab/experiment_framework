import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.performance.perf_base import PerfConfig, PerfGraphic


# noinspection PyPep8Naming
class VizScatterRaw(PerfGraphic):

    @property
    def point_size(self):
        # return matplotlib.rcParams['lines.markersize'] ** 2
        return matplotlib.rcParams['lines.markersize']

    @staticmethod
    def feat2_determine(count_feat, count_points, X):
        """
        Based on X it determines what should be the second dimension used for plotting.
        - In the case of 1 feature it uses all zeros but only generates them once using a closure.
        - In the case of 2 dimensions or more the 2nd feature from X is used
        """
        if count_feat >= 2:
            def two_or_more():
                return X[:, 1]

            return two_or_more
        elif count_feat == 1:
            all_zeros = np.zeros(count_points)

            def one():
                return all_zeros

            return one
        else:
            raise Exception(f"Only 0 or negative dimensions not supported but {count_feat} found")

    def _exec(self, exp_result):
        count_points, count_feat = exp_result.dataset.X.shape

        exp = exp_result.exp
        X = exp_result.dataset.X
        feat2 = self.feat2_determine(count_feat, count_points, X)
        point_size = self.point_size

        for i, scores in enumerate(exp_result.trials_scores):
            # Plot True Positive (TP)
            fig = plt.figure(dpi=PerfConfig.VIZ_DPI)
            plt.scatter(X[:, 0], feat2(), s=point_size, marker=PerfConfig.ScatterPlot.MARKER_RAW)

            if PerfConfig.SHOW_HEADINGS:
                plt.title(f'{exp_result} Scatter Plot (Raw)' + ('' if count_feat <= 2 else f', 2 of {count_feat} feat'))
            plt.grid(True)

            if PerfConfig.ScatterPlot.SAME_SCALE:
                x_limits = plt.xlim()
                y_limits = plt.ylim()
                both_limits = x_limits + y_limits
                new_limits = [min(both_limits), max(both_limits)]
                plt.xlim(new_limits[0], new_limits[1])
                plt.ylim(new_limits[0], new_limits[1])

            if PerfConfig.ScatterPlot.EQUAL_ASPECT:
                fig.axes[0].set_aspect('equal', adjustable='box')

            if PerfConfig.VIZ_SAVE_TO_DISK:
                plt.savefig(self.viz_file_name(exp_result.exp, i + 1))
            if PerfConfig.VIZ_DISPLAY:
                plt.show()
            plt.close()
