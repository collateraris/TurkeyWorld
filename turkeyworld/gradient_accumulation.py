"""Streaming scalar means without retaining any microbatch autograd graph."""
from __future__ import annotations

import math
from numbers import Real


class MetricMean:
    def __init__(self):
        self.sums, self.counts, self.flags = {}, {}, {}

    def add(self, values):
        for name, value in values.items():
            if isinstance(value, bool):
                self.flags[name] = self.flags.get(name, True) and value
            elif isinstance(value, Real):
                value = float(value)
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite accumulated metric: '+name)
                self.sums[name] = self.sums.get(name, 0.)+value
                self.counts[name] = self.counts.get(name, 0)+1

    def mean(self):
        return {**{name: value/self.counts[name] for name, value in self.sums.items()}, **self.flags}
