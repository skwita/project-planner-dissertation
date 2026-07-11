"""Topological scheduler that assigns planned and real start/end times."""

from collections import defaultdict, deque

from models.task import Task


def topo_sort_tasks(tasks: list[Task]) -> list[int]:
    """
    Return task IDs in topological order (Kahn's algorithm).

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    graph: dict[int, list[int]] = defaultdict(list)
    in_degree: dict[int, int] = defaultdict(int)

    for task in tasks:
        in_degree.setdefault(task.task_id, 0)
        for dep_id in task.dependencies:
            graph[dep_id].append(task.task_id)
            in_degree[task.task_id] += 1

    queue: deque[int] = deque(
        tid for tid, deg in in_degree.items() if deg == 0
    )
    order: list[int] = []
    while queue:
        tid = queue.popleft()
        order.append(tid)
        for neighbour in graph[tid]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(tasks):
        raise ValueError("Dependency graph contains a cycle.")

    return order


def build_schedule(tasks: list[Task], percentile: float,
                   seed: int | None = None) -> list[Task]:
    """
    Sample task durations and schedule them in dependency order.

    Both *planned* and *real* timelines are computed:
      - Planned: uses the lognormal percentile as the task duration.
      - Real: uses a random lognormal draw; start time is the max of
        the planned start, all predecessors' real end times, and the
        role's real availability.

    Each task gets its own derived seed (``seed + i``) so that results
    are reproducible per-task without tasks interfering with each other's
    RNG state.

    Args:
        tasks:      List of Task objects (mutated in-place).
        percentile: Planning percentile for planned durations (0–1).
        seed:       Master RNG seed; None means fully random.

    Returns:
        The same list of tasks, now with all schedule fields populated.
    """
    # 1. Sample durations — each task gets its own derived seed
    for i, task in enumerate(tasks):
        task.sample_durations(percentile, None if seed is None else seed + i)

    task_map: dict[int, Task] = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    # 2. Assign start/end times in topological order
    role_planned_ready: dict[str, float] = defaultdict(float)
    role_real_ready: dict[str, float] = defaultdict(float)

    for tid in order:
        task = task_map[tid]

        # --- Planned timeline ---
        planned_dep_end = max(
            (task_map[dep].planned_end_time for dep in task.dependencies),
            default=0.0,
        )
        task.planned_start_time = max(planned_dep_end, role_planned_ready[task.role])
        task.planned_end_time = task.planned_start_time + task.planned_duration
        role_planned_ready[task.role] = task.planned_end_time

        # --- Real timeline ---
        real_dep_end = max(
            (task_map[dep].real_end_time for dep in task.dependencies),
            default=0.0,
        )
        task.real_start_time = max(
            task.planned_start_time,
            real_dep_end,
            role_real_ready[task.role],
        )
        task.real_end_time = task.real_start_time + task.real_duration
        role_real_ready[task.role] = task.real_end_time

    return tasks
