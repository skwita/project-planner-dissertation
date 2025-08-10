from collections import defaultdict, deque
import numpy as np

def build_schedule(tasks, percentile, seed=None):
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
    # earliest_start = defaultdict(float)
    # role_available_time = defaultdict(float)  # Когда роль освободится
    
    # for task_id in scheduled_order:
    #     task = task_map[task_id]
        
    #     # Максимум из:
    #     # - времени завершения всех зависимостей
    #     # - времени доступности роли
        
    #     dep_end_time = max(
    #         [earliest_start[dep_id] + max(task_map[dep_id].planned_duration, task_map[dep_id].real_duration) 
    #          for dep_id in task.dependencies],
    #         default=0
    #     )
        
    #     task.start_time = max(dep_end_time, role_available_time[task.role])
    #     earliest_start[task_id] = task.start_time

    #     planned_dependency_end = max(
    #         [task_map[dep].start_time + task_map[dep].planned_duration for dep in task.dependencies],
    #         default=0
    #     )

    #     task.planned_start_time = max(planned_dependency_end, role_available_time[task.role])
        
    #     # Обновляем время доступности роли
        
    #     role_available_time[task.role] = task.start_time + max(task.planned_duration, task.real_duration)
    
    # return tasks

    role_planned_ready = defaultdict(float)
    role_real_ready = defaultdict(float)

    for task_id in scheduled_order:
        task = task_map[task_id]

        # === Плановое выполнение ===

        # Плановое завершение всех предшественников
        planned_dep_end = max(
            [task_map[dep].planned_start_time + task_map[dep].planned_duration for dep in task.dependencies],
            default=0
        )

        # Плановое начало = максимум из планового конца зависимостей и плановой готовности ресурса
        task.planned_start_time = max(planned_dep_end, role_planned_ready[task.role])
        task.planned_end_time = task.planned_start_time + task.planned_duration

        # Обновляем, когда роль будет готова по плану
        role_planned_ready[task.role] = task.planned_end_time

        # === Фактическое выполнение ===

        # Фактическое завершение всех предшественников
        real_dep_end = max(
            [task_map[dep].real_end_time for dep in task.dependencies],
            default=0
        )

        # Реальная готовность роли
        resource_ready = role_real_ready[task.role]

        # Фактическое начало = макс(плановое начало, конец предшественников, доступность ресурса)
        task.real_start_time = max(task.planned_start_time, real_dep_end, resource_ready)
        task.real_end_time = task.real_start_time + task.real_duration

        # Обновляем, когда роль снова будет доступна
        role_real_ready[task.role] = task.real_end_time

    return tasks