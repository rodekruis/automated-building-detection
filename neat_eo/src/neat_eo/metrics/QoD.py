import torch
import math

from neat_eo.metrics.core import confusion


def get(label, mask, config=None):

    tn, fn, fp, tp = confusion(label, mask)

    try:
        iou = tp / (tp + fn + fp)
    except ZeroDivisionError:
        iou = float("NaN")

    W, H = mask.size()
    ratio = float(100 * torch.max(torch.sum(mask.float()), torch.sum(label.float())) / (W * H))
    dist = 0.0 if iou != iou else 1.0 - iou

    qod = 100 - (dist * (math.log(ratio + 1.0) + 1e-7) * (100 / math.log(100)))
    qod = 0.0 if qod < 0.0 else qod  # Corner case prophilaxy

    return qod / 100.0
