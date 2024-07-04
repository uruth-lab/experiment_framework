import matplotlib.pyplot as plt
from sklearn import metrics

from src.performance.perf_base import PerfConfig, PerfGraphic


class VizPrecisionRecall(PerfGraphic):
    def _exec(self, exp_result):
        lw = PerfConfig.VIZ_LINE_WEIGHT
        i = 1
        for scores in exp_result.trials_scores:
            precisions, recalls, thresholds = \
                metrics.precision_recall_curve(exp_result.dataset.y, -scores)
            auc = metrics.auc(recalls, precisions)
            plt.figure(dpi=PerfConfig.VIZ_DPI)
            plt.plot(recalls, precisions, color='darkgreen',
                     lw=lw,
                     label='Precision-Recall curve (area = %0.2f)' % auc)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            if PerfConfig.SHOW_HEADINGS:
                plt.title(f'{exp_result} Precision-Recall Curve')
            plt.legend(loc="lower left")
            if PerfConfig.VIZ_SAVE_TO_DISK:
                plt.savefig(self.viz_file_name(exp_result.exp, i))
            if PerfConfig.VIZ_DISPLAY:
                plt.show()
            i += 1
            plt.close()
