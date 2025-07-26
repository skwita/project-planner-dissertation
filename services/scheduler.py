from collections import defaultdict, deque
import numpy as np

def build_schedule(tasks, percentile, with_buffer=True, seed=None):
    # 1. Генерация длительностей задач
    for task in tasks:
        task.sample_durations(percentile, seed)
    
    # 2. Построение графа зависимостей
    task_map = {t.task_id: t for t in tasks}
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for task in tasks:
        for dep_id in task.dependencies:
            graph[dep_id].append(task.task_id)
            in_degree[task.task_id] += 1
    
    # 3. Топологическая сортировка (Kahn's algorithm)
    queue = deque()
    for task in tasks:
        if in_degree[task.task_id] == 0:
            queue.append(task.task_id)
    
    scheduled_order = []
    while queue:
        task_id = queue.popleft()
        scheduled_order.append(task_id)
        
        for neighbor in graph[task_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 4. Расчет временных параметров
    earliest_start = defaultdict(float)
    role_available_time = defaultdict(float)  # Когда роль освободится
    
    for task_id in scheduled_order:
        task = task_map[task_id]
        
        # Максимум из:
        # - времени завершения всех зависимостей
        # - времени доступности роли
        if with_buffer:
            dep_end_time = max(
                [earliest_start[dep_id] + max(task_map[dep_id].planned_duration, task_map[dep_id].real_duration) 
                 for dep_id in task.dependencies],
                default=0
            )
        else:
            dep_end_time = max(
                [earliest_start[dep_id] + task_map[dep_id].real_duration
                 for dep_id in task.dependencies],
                default=0
            )
        
        task.start_time = max(dep_end_time, role_available_time[task.role])
        earliest_start[task_id] = task.start_time
        
        # Обновляем время доступности роли
        if with_buffer:
            role_available_time[task.role] = task.start_time + max(task.planned_duration, task.real_duration)
        else:
            role_available_time[task.role] = task.start_time + task.real_duration

    
    return tasks