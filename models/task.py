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
        """
        Расчитывает плановые и фактические начала, длительности и концы задач
        """
        np.random.seed(seed)

        a = 1 + (self.stddev / self.mean) ** 2
        s = np.sqrt(np.log(a))
        scale = self.mean / np.sqrt(a)
        
        self.planned_duration = lognorm.ppf(percentile, s=s, scale=scale)
        self.real_duration = lognorm.rvs(s=s, scale=scale)

    def reset(self):
        self.planned_duration = None
        self.planned_start_time = None
        self.planned_end_time = None
        self.real_duration = None
        self.real_start_time = None
        self.real_end_time = None
