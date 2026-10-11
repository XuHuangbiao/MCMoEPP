import numpy as np


def _safe_corr(preds, labels):
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if preds.size < 2 or np.std(preds) == 0 or np.std(labels) == 0:
        return 0.0
    return float(np.corrcoef(preds, labels)[0, 1])


def _weighted_f1(pred_bin, label_bin):
    scores = []
    weights = []
    for cls in [False, True]:
        pred_cls = pred_bin == cls
        label_cls = label_bin == cls
        tp = float(np.sum(pred_cls & label_cls))
        fp = float(np.sum(pred_cls & ~label_cls))
        fn = float(np.sum(~pred_cls & label_cls))
        support = float(np.sum(label_cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
        weights.append(support)
    total = sum(weights)
    return float(sum(s * w for s, w in zip(scores, weights)) / total) if total > 0 else 0.0


def compute_waf1_acc(pred, label):
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)

    pred_has0 = pred >= 0
    label_has0 = label >= 0
    has0 = {
        "acc": float(np.mean(pred_has0 == label_has0)),
        "waf1": _weighted_f1(pred_has0, label_has0),
    }

    keep = label != 0
    if keep.sum() == 0:
        non0 = {"acc": None, "waf1": None}
    else:
        pred_non0 = pred[keep] > 0
        label_non0 = label[keep] > 0
        non0 = {
            "acc": float(np.mean(pred_non0 == label_non0)),
            "waf1": _weighted_f1(pred_non0, label_non0),
        }
    return has0, non0


def compute_mae_corr_mse(pred, label):
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)
    return {
        "mae": float(np.mean(np.abs(pred - label))),
        "corr": _safe_corr(pred, label),
        "mse": float(np.mean((pred - label) ** 2)),
    }
