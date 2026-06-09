import numpy as np
from sklearn import metrics


def compute_all_metrics(conf, label, recall=0.95):
    """
    Compute OOD detection metrics from confidence scores.

    Args:
        conf (np.ndarray): confidence where larger => more ID-like
                           (e.g., MSP, max softmax prob, -energy).
        label (np.ndarray): binary labels with OOD=1, ID=0.
        recall (float): target TPR level for FPR@TPR (default 0.95).

    Returns:
        dict with keys: fpr95, auroc, aupr_in, aupr_out
    """
    auroc, aupr_in, aupr_out, fpr95 = auc_and_fpr_recall(conf, label, recall)
    return {
        "fpr95": fpr95,
        "auroc": auroc,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
    }


def auc_and_fpr_recall(conf, label, tpr_th=0.95):
    """
    OOD is the positive class (label==1).

    Assumption: ID samples have larger 'conf' than OOD samples.
    Therefore, we use -conf as the score for ROC where higher => more OOD-like.
    """

    label = np.asarray(label).astype(np.int32)
    conf = np.asarray(conf).astype(np.float64)

    # ROC: need higher score => more positive (OOD). Use -conf.
    fpr_list, tpr_list, _ = metrics.roc_curve(label, -conf)
    idx = np.argmax(tpr_list >= tpr_th)
    fpr = fpr_list[idx] if idx < len(fpr_list) else 1.0

    # PR curves
    precision_in, recall_in, _ = metrics.precision_recall_curve(1 - label, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(label, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, float(fpr)

