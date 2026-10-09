"""Dependency-free paired statistics for repeated policy experiments."""
from __future__ import annotations

import itertools
import math
from statistics import mean, stdev


def paired_effect(values_a, values_b):
    """Return mean difference, SD, Cohen's dz, and exact sign-flip p-value."""
    diffs = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "sd_diff": 0.0, "cohens_dz": 0.0, "p_value": 1.0}
    md = mean(diffs)
    sd = stdev(diffs) if n > 1 else 0.0
    dz = md / sd if sd > 0 else (math.inf if md > 0 else (-math.inf if md < 0 else 0.0))

    # Exact paired randomization test.  For n>15, use deterministic Monte
    # Carlo sign flips to avoid exponential growth.
    observed = abs(md)
    if n <= 15:
        signed_means = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            signed_means.append(abs(mean(d * s for d, s in zip(diffs, signs))))
        p = sum(x >= observed - 1e-15 for x in signed_means) / len(signed_means)
    else:
        # Deterministic pseudo-random sign sequence.
        state = 0x12345678
        extreme = 0
        total = 20000
        for _ in range(total):
            signed = 0.0
            for d in diffs:
                state = (1103515245 * state + 12345) & 0x7FFFFFFF
                signed += d if (state & 1) else -d
            if abs(signed / n) >= observed - 1e-15:
                extreme += 1
        p = (extreme + 1) / (total + 1)

    return {
        "mean_diff": md,
        "sd_diff": sd,
        "cohens_dz": dz,
        "p_value": p,
    }
