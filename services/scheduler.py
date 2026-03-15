from collections import defaultdict, deque


def topo_sort_tasks(tasks):
    task_map = {t.task_id: t for t in tasks}
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for task in tasks:
        in_degree.setdefault(task.task_id, 0)

    for task in tasks:
        for dep_id in task.dependencies:
            graph[dep_id].append(task.task_id)
            in_degree[task.task_id] += 1

    queue = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
    order = []

    while queue:
        task_id = queue.popleft()
        order.append(task_id)

        for neighbor in graph[task_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(tasks):
        raise ValueError("В графе задач обнаружен цикл")

    return order


def build_schedule(tasks, percentile, seed=None):
    """
    Исходное построение плана + одной реализации факта.
    """
    for i, task in enumerate(tasks):
        task.sample_durations(percentile, None if seed is None else seed + i)

    task_map = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    role_planned_ready = defaultdict(float)
    role_real_ready = defaultdict(float)

    for task_id in order:
        task = task_map[task_id]

        planned_dep_end = max(
            [task_map[dep].planned_end_time for dep in task.dependencies],
            default=0.0
        )
        task.planned_start_time = max(planned_dep_end, role_planned_ready[task.role])
        task.planned_end_time = task.planned_start_time + task.planned_duration
        role_planned_ready[task.role] = task.planned_end_time

        real_dep_end = max(
            [task_map[dep].real_end_time for dep in task.dependencies],
            default=0.0
        )
        task.real_start_time = max(task.planned_start_time, real_dep_end, role_real_ready[task.role])
        task.real_end_time = task.real_start_time + task.real_duration
        role_real_ready[task.role] = task.real_end_time

    return tasks