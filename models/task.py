import numpy as np
from scipy.stats import lognorm

class Task:
    def __init__(self, task_id, role, dependencies, mean, stddev):
        self.task_id = int(task_id)
        self.role = role
        self.dependencies = dependencies
        self.mean = float(mean)
        self.stddev = float(stddev)
        self.planned_duration = None
        self.real_duration = None
        self.start_time = None
        self.end_time = None

    def sample_durations(self, percentile, seed=None):
        np.random.seed(seed)
        sigma = self.stddev
        mu = np.log(self.mean) - 0.5 * sigma ** 2
        self.planned_duration = lognorm.ppf(percentile, sigma, scale=np.exp(mu))
        self.real_duration = lognorm.rvs(sigma, scale=np.exp(mu))
