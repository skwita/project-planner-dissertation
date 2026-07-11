"""Critical path and critical chain algorithms."""

from collections import defaultdict, deque

from models.task import Task


# ---------------------------------------------------------------------------
# Critical path (dependency graph only, based on mean durations)
# ---------------------------------------------------------------------------

def find_critical_path(tasks: list[Task]) -> list[Task]:
    """
    Find the critical path using forward/backward pass on mean durations.

    The critical path is the longest sequence of dependent tasks from
    project start to finish (ignoring resource contention).

    Returns:
        Ordered list of Task objects on the critical path.
    """
    task_by_id = {t.task_id: t for t in tasks}

    # Build graph and compute in-degrees
    graph: dict[int, list[int]] = defaultdict(list)
    in_degree: dict[int, int] = defaultdict(int)
    for task in tasks:
        in_degree.setdefault(task.task_id, 0)
        for dep in task.dependencies:
            graph[dep].append(task.task_id)
            in_degree[task.task_id] += 1

    # Topological sort (Kahn's)
    queue: deque[int] = deque(
        tid for tid, deg in in_degree.items() if deg == 0
    )
    topo_order: list[int] = []
    while queue:
        tid = queue.popleft()
        topo_order.append(tid)
        for nxt in graph[tid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Forward pass: earliest start/finish
    earliest_start: dict[int, float] = {t.task_id: 0.0 for t in tasks}
    earliest_finish: dict[int, float] = {}
    for tid in topo_order:
        t = task_by_id[tid]
        start = max((earliest_finish[d] for d in t.dependencies), default=0.0)
        earliest_start[tid] = start
        earliest_finish[tid] = start + t.mean

    # Backward pass: identify critical path by tracing from the last task
    project_end = max(earliest_finish.values())
    end_tid = next(
        tid for tid, finish in earliest_finish.items() if finish == project_end
    )

    critical_path: list[Task] = []
    current = end_tid
    while True:
        critical_path.append(task_by_id[current])
        preds = [
            d for d in task_by_id[current].dependencies
            if abs(earliest_finish[d] - earliest_start[current]) < 1e-6
        ]
        if not preds:
            break
        current = preds[0]

    critical_path.reverse()
    return critical_path


# ---------------------------------------------------------------------------
# Critical chain (dependency graph + resource sequencing)
# ---------------------------------------------------------------------------

def critical_chain_path(tasks: list[Task]) -> list[Task]:
    """
    Build the critical chain: the longest path accounting for both
    task dependencies and resource (role) contention.

    If the combined graph contains a cycle (e.g. conflicting planned
    times), resource edges are removed one by one until the graph is
    acyclic (simple greedy heuristic).

    Returns:
        Ordered list of Task objects along the critical chain.

    Raises:
        RuntimeError: If the graph cannot be made acyclic.
    """
    task_by_id = {t.task_id: t for t in tasks}

    # Effective duration: prefer scheduled times, fall back to mean
    def _duration(t: Task) -> float:
        if t.planned_start_time is not None and t.planned_end_time is not None:
            return float(t.planned_end_time - t.planned_start_time)
        return float(t.mean)

    duration = {t.task_id: _duration(t) for t in tasks}

    # ---- Build graph (dependency edges) ----
    graph: dict[int, list[int]] = defaultdict(list)
    in_deg: dict[int, int] = defaultdict(int)
    for t in tasks:
        in_deg.setdefault(t.task_id, 0)
    for t in tasks:
        for dep in t.dependencies or []:
            if dep not in task_by_id:
                continue
            graph[dep].append(t.task_id)
            in_deg[t.task_id] += 1

    # ---- Add resource-sequencing edges ----
    def _sort_key(t: Task) -> tuple:
        if t.planned_start_time is not None:
            return (0, float(t.planned_start_time))
        if t.planned_end_time is not None:
            return (1, float(t.planned_end_time))
        return (2, int(t.task_id))

    role_groups: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        role_groups[t.role].append(t)

    resource_edges: set[tuple[int, int]] = set()
    for role_tasks in role_groups.values():
        ordered = sorted(role_tasks, key=_sort_key)
        for a, b in zip(ordered, ordered[1:]):
            graph[a.task_id].append(b.task_id)
            in_deg[b.task_id] += 1
            resource_edges.add((a.task_id, b.task_id))

    # ---- Topological sort (with cycle-breaking fallback) ----
    def _topo(g: dict, deg: dict) -> list[int]:
        deg = dict(deg)
        q: deque[int] = deque(tid for tid, d in deg.items() if d == 0)
        order: list[int] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in g.get(u, []):
                deg[v] -= 1
                if deg[v] == 0:
                    q.append(v)
        return order

    topo = _topo(graph, in_deg)
    if len(topo) != len(tasks):
        # Remove resource edges one by one until acyclic
        for a, b in list(resource_edges):
            if len(topo) == len(tasks):
                break
            if b in graph.get(a, []):
                graph[a].remove(b)
                in_deg[b] -= 1
                topo = _topo(graph, in_deg)
        if len(topo) != len(tasks):
            raise RuntimeError(
                "Cannot make the dependency graph acyclic. "
                "Check task dependencies and planned times."
            )

    # ---- Longest-path (critical chain) ----
    dist: dict[int, float] = {t.task_id: float("-inf") for t in tasks}
    prev: dict[int, int | None] = {t.task_id: None for t in tasks}

    for tid in topo:
        if dist[tid] == float("-inf"):
            dist[tid] = duration[tid]
    for u in topo:
        if dist[u] == float("-inf"):
            dist[u] = duration[u]
        for v in graph.get(u, []):
            candidate = dist[u] + duration[v]
            if candidate > dist[v]:
                dist[v] = candidate
                prev[v] = u

    # Reconstruct path from end
    end_tid = max(dist, key=lambda k: dist[k])
    path_ids: list[int] = []
    cur: int | None = end_tid
    while cur is not None:
        path_ids.append(cur)
        cur = prev[cur]
    path_ids.reverse()
    return [task_by_id[i] for i in path_ids]
