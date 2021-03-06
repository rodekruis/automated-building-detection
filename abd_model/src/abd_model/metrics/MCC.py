import math
from abd_model.metrics.core import confusion


def get(label, predicted, config=None):

    tn, fn, fp, tp = confusion(label, predicted)
    if tp == 0 and fp == 0 and fn == 0:
        return float("NaN")

    try:
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except ZeroDivisionError:
        mcc = float("NaN")

    return mcc
