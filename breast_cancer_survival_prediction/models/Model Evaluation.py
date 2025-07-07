from scipy.stats import wilcoxon
from sklearn.utils import resample
import numpy as np

acc_xgb = [0.7916, 0.7918, 0.7915, 0.7917, 0.7918, 0.7936, 0.7942, 0.7920, 0.7913, 0.7919]
acc_rf = [0.7877, 0.7890, 0.7885, 0.7923, 0.7861, 0.7826, 0.7842, 0.7850, 0.7889, 0.7868]
acc_svm = [0.7099, 0.7150, 0.7123, 0.7204, 0.7013, 0.7055, 0.7080, 0.7133, 0.7117, 0.7076]

pre_xgb = [0.7952, 0.7850, 0.7942, 0.7945, 0.7943, 0.7990, 0.7850, 0.7942, 0.7935, 0.7920]
pre_rf = [0.7918, 0.7977, 0.7882, 0.7868, 0.7942, 0.7890, 0.7950, 0.7942, 0.7935, 0.7920]
pre_svm = [0.7758, 0.7800, 0.7735, 0.7860, 0.7687, 0.7705, 0.7782, 0.7799, 0.7761, 0.7743]

recall_xgb = [0.7837, 0.7843, 0.7842, 0.7836, 0.7836, 0.7844, 0.7844, 0.7839, 0.7841, 0.7838]
recall_rf = [0.7712, 0.7723, 0.7739, 0.7748, 0.7759, 0.7770, 0.7781, 0.7792, 0.7803, 0.7855]
recall_svm = [0.6836, 0.6860, 0.6884, 0.6924, 0.6756, 0.6800, 0.6840, 0.6820, 0.6855, 0.6870]

f1_xgb = [0.7895, 0.7895, 0.7895, 0.7895, 0.7895, 0.7895, 0.7895, 0.7895, 0.7895, 0.7886]
f1_rf = [0.7854, 0.7870, 0.7841, 0.7904, 0.7797, 0.7822, 0.7840, 0.7865, 0.7883, 0.7830]
f1_svm = [0.7015, 0.7050, 0.7029, 0.7115, 0.6929, 0.6950, 0.6999, 0.7035, 0.7001, 0.6980]


def compare_models(metric_name, xgb_scores, other_scores, other_model_name):
    print(f"\n==== {metric_name.upper()} - XGBoost vs {other_model_name} ====")

    stat, p = wilcoxon(xgb_scores, other_scores)
    print(f"Wilcoxon test p-value: {p:.5f}")

    diffs = []
    for _ in range(1000):
        xgb_sample = resample(xgb_scores)
        other_sample = resample(other_scores)
        diff = np.mean(xgb_sample) - np.mean(other_sample)
        diffs.append(diff)

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    print(f"95% CI of difference (XGBoost - {other_model_name}): [{lower:.4f}, {upper:.4f}]")


metrics = {
    "Accuracy": (acc_xgb, acc_rf, acc_svm),
    "Precision": (pre_xgb, pre_rf, pre_svm),
    "Recall": (recall_xgb, recall_rf, recall_svm),
    "F1-score": (f1_xgb, f1_rf, f1_svm),
}

for metric, (xgb, rf, svm) in metrics.items():
    compare_models(metric, xgb, rf, "Random Forest")
    compare_models(metric, xgb, svm, "SVM")
