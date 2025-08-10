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
        self.planned_start_time = None
        self.planned_end_time = None
        self.real_duration = None
        self.real_start_time = None
        self.real_end_time = None

    def sample_durations(self, percentile, seed=None):
        np.random.seed(seed)
        # sigma = self.stddev

        a = 1 + (self.stddev / self.mean) ** 2
        s = np.sqrt(np.log(a))
        scale = self.mean / np.sqrt(a)
        
        self.planned_duration = lognorm.ppf(percentile, s=s, scale=scale)
        self.real_duration = lognorm.rvs(s=s, scale=scale)

        # sigma = np.sqrt(np.log(1 + np.sqrt(self.stddev / self.mean)))
        # mu = np.log(self.mean) - 0.5 * sigma ** 2
        # self.planned_duration = lognorm.ppf(percentile, s=sigma, scale=np.exp(mu))
        # self.real_duration = lognorm.rvs(s=sigma, scale=np.exp(mu))
