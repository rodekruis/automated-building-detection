from neat_eo.metrics.core import confusion


def get(label, predicted, config=None):

    tn, fn, fp, tp = confusion(label, predicted)
    if tp == 0 and fp == 0 and fn == 0:
        return float("NaN")

    try:
        assert tp or fp or fn
        iou = float(tp / (fp + fn + tp))
    except ZeroDivisionError:
        iou = float("NaN")

    return iou
