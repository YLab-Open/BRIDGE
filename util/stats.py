from __future__ import annotations

import numpy as np
from scipy import stats


def get_mean_std_ci(data, confidence=0.95):
    """Return mean, sample std, and t confidence interval for numeric data."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, std, [mean - h, mean + h]
