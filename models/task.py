"""Task model with lognormal duration sampling."""

import numpy as np
from scipy.stats import lognorm

from utils.lognormal import lognorm_scipy_params, lognorm_ppf, lognorm_rvs


class Task:
    """
    Represents a project task with planned and actual schedule times.

    Duration is modelled as lognormal, parameterised by mean and stddev.
    """

    def __init__(self, task_id: int, role: str, dependencies: list[int],
                 mean: float, stddev: float):
        self.task_id = int(task_id)
        self.role = role
        self.dependencies = dependencies
        self.mean = float(mean)
        self.stddev = float(stddev)

        # Set by build_schedule / sample_durations
        self.planned_duration: float | None = None
        self.planned_start_time: float | None = None
        self.planned_end_time: float | None = None
        self.real_duration: float | None = None
        self.real_start_time: float | None = None
        self.real_end_time: float | None = None

    def sample_durations(self, percentile: float, seed: int | None = None) -> None:
        """
        Draw planned (deterministic percentile) and real (random) durations.

        Args:
            percentile: task planning percentile, e.g. 0.5 for the median.
            seed: RNG seed for reproducibility of the real duration draw.
        """
        self.planned_duration = lognorm_ppf(percentile, self.mean, self.stddev)
        rng = np.random.default_rng(seed)
        self.real_duration = lognorm_rvs(self.mean, self.stddev, rng)

    def reset(self) -> None:
        """Clear all scheduled times so the task can be re-scheduled."""
        self.planned_duration = None
        self.planned_start_time = None
        self.planned_end_time = None
        self.real_duration = None
        self.real_start_time = None
        self.real_end_time = None

    def __repr__(self) -> str:
        return (f"Task(id={self.task_id}, role={self.role!r}, "
                f"mean={self.mean}, stddev={self.stddev})")
