import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from src.performance.perf_base import PerfConfig, PerfGraphic


class VizROC(PerfGraphic):
    def _exec(self, exp_result):
        lw = PerfConfig.VIZ_LINE_WEIGHT
        i = 1
        for scores in exp_result.trials_scores:
            fpr, tpr, _ = roc_curve(exp_result.dataset.y, -scores)
            roc_auc = auc(fpr, tpr)
            plt.figure(dpi=PerfConfig.VIZ_DPI)
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw,
                     label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], label='Chance', color='navy', lw=lw,
                     linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            if PerfConfig.SHOW_HEADINGS:
                plt.title(f'{exp_result} Receiver operating characteristic')
            plt.legend(loc="lower right")
            if PerfConfig.VIZ_SAVE_TO_DISK:
                plt.savefig(self.viz_file_name(exp_result.exp, i))
            if PerfConfig.VIZ_DISPLAY:
                plt.show()
            i += 1
            plt.close()
