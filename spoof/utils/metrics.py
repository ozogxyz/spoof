import numpy as np
from sklearn.metrics import roc_curve


EPS = 1e-15
LABEL_LIVE = 1
LABEL_FAKE = 0


def accuracy(labels, predicted, th=0.5):
    predicted = np.int32(predicted > th)
    labels = np.int32(labels)
    return np.float32(labels == predicted).mean().item()


def fcl(labels, scores, th=0.5):
    """
    Fake Called Live OR APCER (Attack Presentation Classification Error Rate)
    """
    labels = np.int32(labels)
    predicted = np.int32(np.array(scores) > th)
    n_fake = (labels != LABEL_LIVE).sum()
    fake_called_live = (predicted == LABEL_LIVE) & (labels != LABEL_LIVE)
    return (fake_called_live.sum() / (n_fake + EPS)).item()


def lcf(labels, scores, th=0.5):
    """
    Live Called Fake OR BPCER (Bona-fide Presentation Classification Acceptance Rate )
    """
    labels = np.int32(labels)
    predicted = np.int32(np.array(scores) > th)
    n_live = (labels == LABEL_LIVE).sum()
    live_called_fake = (predicted != LABEL_LIVE) & (labels == LABEL_LIVE)
    return (live_called_fake.sum() / (n_live + EPS)).item()


def eer(labels, scores):
    """
    Equal Error Rate with the corresponding threshold
    """
    labels = np.int32(labels)
    scores = np.float32(scores)
    fake_preds = scores[labels == LABEL_FAKE]
    live_preds = scores[labels == LABEL_LIVE]
    if len(live_preds) == 0 or len(fake_preds) == 0:
        return -1.0, -1.0
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    fnr = 1 - tpr
    thresh = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_val = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer_val.item(), thresh.item()
