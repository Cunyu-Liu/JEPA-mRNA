import numpy as np

from eval.ss.metrics import StructureLevelMetrics


LENGTH_BUCKETS = ("<=100", "100-200", "200-400", ">400")


def length_bucket(length):
    if length <= 100:
        return 0
    if length <= 200:
        return 1
    if length <= 400:
        return 2
    return 3


def legal_probabilities(scores, mask):
    selection = np.triu(np.asarray(mask, dtype=bool), k=1)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))
    return probabilities[selection]


def structure_confidence(scores, pairs):
    if not pairs:
        return 0.0
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))
    return float(np.exp(np.mean(np.log(np.clip(
        [probabilities[i, j] for i, j in pairs], 1e-12, 1.0)))))


def inf_value(pred_pairs, gt_pairs):
    return StructureLevelMetrics.from_pairs(pred_pairs, gt_pairs).inf


class PairCalibrationAccumulator:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.count = np.zeros(n_bins, dtype=np.int64)
        self.sum_probability = np.zeros(n_bins, dtype=np.float64)
        self.sum_label = np.zeros(n_bins, dtype=np.float64)
        self.sum_squared_error = 0.0
        self.sum_nll = 0.0

    def add(self, scores, mask, gt_pairs):
        selection = np.triu(np.asarray(mask, dtype=bool), k=1)
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))
        p = probabilities[selection]
        labels = np.zeros(selection.shape, dtype=np.float64)
        for i, j in gt_pairs:
            if 0 <= i < len(labels) and 0 <= j < len(labels) and selection[i, j]:
                labels[i, j] = 1.0
        y = labels[selection]
        indices = np.clip((p * self.n_bins).astype(int), 0, self.n_bins - 1)
        self.count += np.bincount(indices, minlength=self.n_bins)
        self.sum_probability += np.bincount(indices, weights=p, minlength=self.n_bins)
        self.sum_label += np.bincount(indices, weights=y, minlength=self.n_bins)
        self.sum_squared_error += float(np.sum((p - y) ** 2))
        clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        self.sum_nll += float(-np.sum(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))

    def finalize(self):
        n = int(self.count.sum())
        if not n:
            return {"n": 0, "n_bins": self.n_bins, "ece": 0.0, "brier": 0.0, "nll": 0.0,
                    "mean_predicted": 0.0, "mean_observed": 0.0,
                    "marginal_calibration_error": 0.0, "reliability": []}
        reliability = []
        ece = 0.0
        for index in range(self.n_bins):
            count = int(self.count[index])
            if not count:
                continue
            predicted = float(self.sum_probability[index] / count)
            observed = float(self.sum_label[index] / count)
            ece += count / n * abs(predicted - observed)
            reliability.append({"bin": index, "lo": index / self.n_bins,
                                "hi": (index + 1) / self.n_bins, "count": count,
                                "mean_predicted": predicted, "fraction_positive": observed})
        mean_predicted = float(self.sum_probability.sum() / n)
        mean_observed = float(self.sum_label.sum() / n)
        return {"n": n, "n_bins": self.n_bins, "ece": float(ece),
                "brier": self.sum_squared_error / n, "nll": self.sum_nll / n,
                "mean_predicted": mean_predicted, "mean_observed": mean_observed,
                "marginal_calibration_error": abs(mean_predicted - mean_observed),
                "reliability": reliability}
