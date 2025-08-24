from concurrent import futures
from collections import defaultdict
import numpy as np
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from tqdm import tqdm
import copy

def calculate_idle_time_old(tasks):
    """
    Возвращает словарь: роль → суммарный простой (в человеко-днях)
    """
    role_tasks = defaultdict(list)

    for task in tasks:
        role_tasks[task.role].append(task)

    role_idle_time = {}

    for role, tasks_for_role in role_tasks.items():
        # Сортируем задачи по началу выполнения
        sorted_tasks = sorted(tasks_for_role, key=lambda t: t.real_start_time)

        idle = 0.0

        for task in sorted_tasks:
            idle += task.real_start_time - task.planned_start_time

        role_idle_time[role] = idle

    return role_idle_time

def calculate_idle_time(tasks):
    """
    Возвращает словарь: роль → суммарный простой (в человеко-днях).
    Простой учитывается только если основная причина задержки (последний завершившийся предшественник)
    принадлежит другой роли.
    """
    role_idle_time = defaultdict(float)

    # словарь для доступа по id
    task_by_id = {task.task_id: task for task in tasks}

    for task in tasks:
        if not task.dependencies:
            continue  # задача без предшественников не ждёт никого

        delay = task.real_start_time - task.planned_start_time
        if delay <= 0:
            continue  # нет задержки → нет простоя

        # Находим предшественника, который завершился последним
        latest_pred = max(
            (task_by_id[dep_id] for dep_id in task.dependencies if dep_id in task_by_id),
            key=lambda t: t.real_end_time,
            default=None
        )

        if latest_pred is None:
            continue

        # Если задержка из-за другой роли → считаем простой
        if latest_pred.role != task.role:
            role_idle_time[task.role] += delay

    return dict(role_idle_time)

def calculate_project_duration(tasks):
    return max(task.real_start_time + task.real_duration for task in tasks)


def monte_carlo_simulation(task_file, percentile, n_iter, seed):
    """
    Выполняет n_iter симуляций для заданного процентиля
    Возвращает массив из длительностей рассчитанных проектов
    """
    rng = np.random.default_rng(seed)

    # Загружаем задачи один раз
    base_tasks = load_tasks_from_csv(task_file)

    # Предгенерация seed-ов для итераций
    seeds = rng.integers(1_000_000, size=n_iter)

    durations = np.empty(n_iter, dtype=float)   # быстрее чем list
    idle_records = []

    for i, sim_seed in enumerate(tqdm(seeds, desc=f"Процентиль {percentile}", leave=False)):
        # Копируем задачи (без повторного чтения файла)
        for task in base_tasks:
            task.reset()

        # Строим расписание
        build_schedule(base_tasks, percentile=percentile, seed=sim_seed)

        # Длительность проекта = max(real_end_time)
        duration = max(t.real_end_time for t in base_tasks)

        # Простой по ролям
        idle = calculate_idle_time(base_tasks)

        durations[i] = duration
        idle_records.append(idle)

    return durations, idle_records

def parallel_monte_carlo_simulation(task_file, percentiles, n_iter, seed):
    results = {}
    with futures.ProcessPoolExecutor() as executor:
        # отправляем задачи и запоминаем, какому p соответствует future
        future_to_p = {
            executor.submit(monte_carlo_simulation, task_file, p, n_iter, seed): p
            for p in percentiles
        }

        # ждём выполнения
        for future in futures.as_completed(future_to_p):
            p = future_to_p[future]
            try:
                results[p] = future.result()
            except Exception as e:
                results[p] = e  # чтобы не терять ошибки
    return results

def calculate_buffer(durations, planned_duration, percentile_project):
    """
    Расчитывает длину буфера проекта при определенном процентиле
    """
    durations = np.array(durations)
    overruns = np.maximum(0, durations - planned_duration)
    buffer_value = np.percentile(sorted(overruns), percentile_project)
    return buffer_value