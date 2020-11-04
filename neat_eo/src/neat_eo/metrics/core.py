import math
import torch
from neat_eo.core import load_module


class Metrics:
    def __init__(self, metrics, classes, config=None):
        self.config = config
        self.classes = classes
        self.metrics = []
        for classe in classes:
            self.metrics.append({metric: [] for metric in metrics})
        self.modules = {metric: load_module("neat_eo.metrics." + metric) for metric in metrics}
        self.n = 0

    def add(self, label, output):
        assert self.modules
        assert self.metrics
        self.n += 1
        for metric, module in self.modules.items():
            for c, classe in enumerate(self.classes):
                mask = (output[c] > 0.5).float()
                self.metrics[c][metric].append(module.get(label, mask, self.config))

    def get(self):
        assert self.metrics

        results = []
        for c, classe in enumerate(self.classes):
            μ = {}
            σ = {}
            for metric, values in self.metrics[c].items():
                n = sum([1 for v in values if not math.isnan(v)])

                try:
                    μ[metric] = sum([v for v in values if not math.isnan(v)]) / n
                except ZeroDivisionError:
                    μ[metric] = float("NaN")

                try:
                    σ[metric] = sum({(math.sqrt(((v - μ[metric]) ** 2))) for v in values if not math.isnan(v)}) / n
                except ZeroDivisionError:
                    σ[metric] = float("NaN")

            results.append({metric: {"μ": μ[metric], "σ": σ[metric]} for metric in self.metrics[c]})

        return results


def confusion(label, predicted):

    confusion = predicted.view(-1).float() / label.view(-1).float()

    tn = torch.sum(torch.isnan(confusion)).item()
    fn = torch.sum(confusion == float("inf")).item()
    fp = torch.sum(confusion == 0).item()
    tp = torch.sum(confusion == 1).item()

    return tn, fn, fp, tp
