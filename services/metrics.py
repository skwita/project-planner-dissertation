"""Project-level metrics: idle time, duration, buffer, and Monte Carlo simulation."""

import copy
from collections import defaultdict
from concurrent import futures

import numpy as np
from tqdm import tqdm

from models.task import Task
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule


# ---------------------------------------------------------------------------
# Single-schedule metrics
# ---------------------------------------------------------------------------

def calculate_project_duration(tasks: list[Task]) -> float:
    """Return the real end time of the last finishing task."""
    return max(t.real_start_time + t.real_duration for t in tasks)


def calculate_idle_time(
    tasks: list[Task],
    baseline_tasks: list[Task] | None = None,
) -> dict[str, float]:
    """
    Compute per-role idle time (in days) caused by cross-role dependencies.

    A role is considered idle when a task's real start is delayed beyond its
    planned start AND the latest-finishing predecessor belongs to a *different*
    role.

    Args:
        tasks:          Scheduled task list with ``real_*`` fields populated.
        baseline_tasks: Optional original plan.  When supplied, the planned
                        start times are taken from here instead of from
                        ``tasks`` itself — needed in replanning contexts where
                        ``planned_*`` fields have been overwritten.  Tasks
                        with ``real_start_time is None`` are skipped.

    Returns:
        Dict mapping role name → cumulative idle days.
    """
    task_by_id = {t.task_id: t for t in tasks}
    baseline_by_id = (
        {t.task_id: t for t in baseline_tasks}
        if baseline_tasks is not None
        else None
    )
    role_idle: dict[str, float] = defaultdict(float)

    for task in tasks:
        if task.real_start_time is None:
            continue
        if not task.dependencies:
            continue

        planned_start = (
            baseline_by_id[task.task_id].planned_start_time
            if baseline_by_id is not None and task.task_id in baseline_by_id
            else task.planned_start_time
        )
        if planned_start is None:
            continue

        # Cross-role idle = how long the task was blocked waiting specifically
        # for predecessors from OTHER roles, measured from the original planned start.
        # We take the latest real_end among cross-role predecessors and compute
        # how much that pushed the start past planned_start.
        # This correctly isolates the cross-role contribution and avoids
        # conflating it with within-role queuing (role_ready delays).
        cross_role_preds = [
            task_by_id[dep]
            for dep in task.dependencies
            if dep in task_by_id
            and task_by_id[dep].role != task.role
            and task_by_id[dep].real_end_time is not None
        ]
        if not cross_role_preds:
            continue

        max_cross_end = max(p.real_end_time for p in cross_role_preds)
        idle = max_cross_end - planned_start
        if idle > 0:
            role_idle[task.role] += idle

    return dict(role_idle)


def calculate_buffer(durations: np.ndarray, planned_duration: float,
                     percentile_project: float) -> float:
    """
    Compute the project buffer at the given confidence level.

    The buffer is the ``percentile_project``-th percentile of schedule
    overruns (i.e. how much the actual duration exceeds the planned one).

    Args:
        durations:         Array of simulated project durations.
        planned_duration:  Planned (deterministic) project duration.
        percentile_project: Confidence level, e.g. 90 for the 90th percentile.

    Returns:
        Buffer size in days.
    """
    overruns = np.maximum(0.0, np.asarray(durations) - planned_duration)
    return float(np.percentile(overruns, percentile_project))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_simulation(
    task_file: str,
    percentile: float,
    n_iter: int,
    seed: int | None,
    max_time: float = 0,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    Run ``n_iter`` Monte Carlo schedule simulations for a single percentile.

    Args:
        task_file:  Path to the tasks CSV.
        percentile: Task planning percentile.
        n_iter:     Number of simulation runs.
        seed:       Master RNG seed (child seeds are derived from it).
        max_time:   Deadline for computing success rate (0 disables it).

    Returns:
        Tuple of (durations, idle_records, success_flags) where:
          - durations:     array of shape (n_iter,) with project durations.
          - idle_records:  list of per-run idle-time dicts.
          - success_flags: array of 0/1 indicating duration ≤ max_time.
    """
    rng = np.random.default_rng(seed)
    base_tasks = load_tasks_from_csv(task_file)
    child_seeds = rng.integers(1_000_000, size=n_iter)

    durations = np.empty(n_iter, dtype=float)
    idle_records: list[dict] = []
    success_flags = np.empty(n_iter, dtype=int)

    for i, sim_seed in enumerate(
        tqdm(child_seeds, desc=f"Percentile {percentile}", leave=False)
    ):
        for task in base_tasks:
            task.reset()

        build_schedule(base_tasks, percentile=percentile, seed=int(sim_seed))

        duration = max(t.real_end_time for t in base_tasks)
        idle = calculate_idle_time(base_tasks)

        durations[i] = duration
        idle_records.append(idle)
        success_flags[i] = 1 if (max_time > 0 and duration <= max_time) else 0

    return durations, idle_records, success_flags


def parallel_monte_carlo_simulation(
    task_file: str,
    percentiles: list[float],
    n_iter: int,
    seed: int | None,
    max_time: float = 0,
) -> dict[float, tuple]:
    """
    Run Monte Carlo simulations for multiple percentiles in parallel.

    Args:
        task_file:   Path to the tasks CSV.
        percentiles: List of task planning percentiles to simulate.
        n_iter:      Iterations per percentile.
        seed:        Master RNG seed.
        max_time:    Deadline for success-rate calculation.

    Returns:
        Dict mapping percentile → (durations, idle_records, success_flags).
    """
    results: dict[float, tuple] = {}
    with futures.ProcessPoolExecutor() as executor:
        future_to_p = {
            executor.submit(
                monte_carlo_simulation, task_file, p, n_iter, seed, max_time
            ): p
            for p in percentiles
        }
        for future in futures.as_completed(future_to_p):
            p = future_to_p[future]
            try:
                results[p] = future.result()
            except Exception as exc:
                results[p] = exc
    return results


def monte_carlo_schedules(
    task_file: str,
    percentile_task: float,
    n_iter: int = 100,
    seed: int | None = None,
) -> list[list[Task]]:
    """
    Return full schedule snapshots (deep-copied task lists) for each run.

    Useful when you need per-run task-level detail rather than summary stats.
    """
    rng = np.random.default_rng(seed)
    base_tasks = load_tasks_from_csv(task_file)
    child_seeds = rng.integers(1_000_000, size=n_iter)
    all_runs: list[list[Task]] = []

    for sim_seed in tqdm(child_seeds, desc=f"Simulating p={percentile_task}", leave=False):
        tasks = copy.deepcopy(base_tasks)
        for task in tasks:
            task.reset()
        build_schedule(tasks, percentile=percentile_task, seed=int(sim_seed))
        all_runs.append(tasks)

    return all_runs
