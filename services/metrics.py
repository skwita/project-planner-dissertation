from collections import defaultdict
import numpy as np
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from tqdm import tqdm

def calculate_idle_time(tasks):
    """
    Возвращает словарь: роль → суммарный простой (в человеко-днях)
    """
    role_tasks = defaultdict(list)

    for task in tasks:
        role_tasks[task.role].append(task)

    role_idle_time = {}

    for role, tasks_for_role in role_tasks.items():
        # Сортируем задачи по началу выполнения
        sorted_tasks = sorted(tasks_for_role, key=lambda t: t.start_time)

        idle = 0.0
        prev_real_end = None

        for task in sorted_tasks:
            if prev_real_end is not None and task.start_time > prev_real_end:
                idle += task.start_time - prev_real_end

            # Обновляем реальное окончание текущей задачи
            prev_real_end = task.start_time + task.real_duration

        role_idle_time[role] = idle

    return role_idle_time

def calculate_project_duration(tasks):
    return max(max(task.start_time + task.real_duration, task.start_time + task.planned_duration) for task in tasks)

def monte_carlo_simulation(task_file, percentile, n_iter, seed, with_buffer):
    """Выполняет n_iter симуляций для заданного процентиля"""

    rng = np.random.default_rng(seed)
    durations = []
    idle_records = []

    for _ in tqdm(range(n_iter), desc=f"Процентиль {percentile}", leave=False):
        tasks = load_tasks_from_csv(task_file)
        tasks = build_schedule(tasks, with_buffer=with_buffer, percentile=percentile, seed=rng.integers(1_000_000))

        duration = max(t.start_time + t.real_duration for t in tasks)
        idle = calculate_idle_time(tasks)

        durations.append(duration)
        idle_records.append(idle)

    return durations, idle_records

