"""
Deadline reachability grid.

For each point in a (project-time-fraction × bias-factor) grid, determines
whether the project deadline can still be met by replanning remaining tasks.

Two grids are computed:
  - ``grid_opt``:  can the deadline be met at ``min_percentile``?
                   (most optimistic possible replanning)
  - ``grid_base``: can the deadline be met at ``base_percentile``?
                   (no change to the planning strategy)

Phase zones on the resulting diagram:
  - Green  (grid_base = True):  deadline reachable without any replanning.
  - Yellow (grid_opt = True, grid_base = False): deadline reachable only if
    we lower the planning percentile.
  - Red    (grid_opt = False):  deadline impossible regardless of replanning.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule, topo_sort_tasks
from utils.lognormal import lognorm_ppf


# ---------------------------------------------------------------------------
# Core forward-scheduling helper
# ---------------------------------------------------------------------------

def _min_finish_biased(
    tasks,
    cutoff_time: float,
    bias: float,
    percentile_remaining: float,
) -> float:
    """
    Compute the earliest achievable project finish time.

    Tasks whose ``planned_end_time`` ≤ ``cutoff_time`` are treated as
    completed with ``actual_duration = planned_duration * bias``.  All
    remaining tasks are scheduled at ``percentile_remaining``.

    Role sequentiality is enforced: same-role tasks queue behind each other.
    Dependency constraints are always respected.

    Args:
        tasks:                 Task list with ``planned_*`` fields set.
        cutoff_time:           Absolute time threshold separating done / remaining.
        bias:                  Actual-to-planned duration ratio for done tasks.
        percentile_remaining:  Planning quantile for not-yet-done tasks.

    Returns:
        Earliest achievable project finish time in the same unit as durations.
    """
    task_map = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    role_ready: dict[str, float] = defaultdict(float)
    actual_end: dict[int, float] = {}

    for tid in order:
        task = task_map[tid]
        dep_end = max(
            (actual_end[d] for d in task.dependencies if d in actual_end),
            default=0.0,
        )
        start = max(dep_end, role_ready[task.role])

        if task.planned_end_time <= cutoff_time:
            duration = task.planned_duration * bias
        else:
            duration = lognorm_ppf(percentile_remaining, task.mean, task.stddev)

        end = start + duration
        actual_end[tid] = end
        role_ready[task.role] = end

    return max(actual_end.values()) if actual_end else 0.0


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def compute_reachability_grid(
    task_file: str,
    time_fractions: np.ndarray,
    bias_values: np.ndarray,
    base_percentile: float = 0.80,
    min_percentile: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build a 2-D reachability grid over project-time fraction and bias factor.

    For each cell ``(i, j)``:
      - ``time_fractions[i]`` — fraction of planned project duration elapsed.
      - ``bias_values[j]``    — ratio actual/planned duration for done tasks.

    Returns two boolean grids of shape ``(len(time_fractions), len(bias_values))``:
      - ``grid_opt[i, j]``  — True if deadline is reachable at ``min_percentile``.
      - ``grid_base[i, j]`` — True if deadline is reachable at ``base_percentile``.

    Args:
        task_file:       Path to the tasks CSV.
        time_fractions:  1-D array of project-time fractions in [0, 1].
        bias_values:     1-D array of bias factors (e.g. ``np.linspace(0.5, 2.5, 20)``).
        base_percentile: Percentile for the baseline plan that defines the deadline.
        min_percentile:  Most optimistic planning percentile for remaining tasks.
        seed:            RNG seed for the baseline ``build_schedule`` call.

    Returns:
        ``(grid_opt, grid_base, planned_duration, deadline)`` where
        ``planned_duration`` is the baseline project length and ``deadline``
        equals ``planned_duration`` (i.e. the deadline is the planned finish).
    """
    tasks = load_tasks_from_csv(task_file)
    build_schedule(tasks, percentile=base_percentile, seed=seed)
    planned_duration = max(t.planned_end_time for t in tasks)
    deadline = planned_duration

    n_t = len(time_fractions)
    n_b = len(bias_values)
    grid_opt  = np.zeros((n_t, n_b), dtype=bool)
    grid_base = np.zeros((n_t, n_b), dtype=bool)

    for i, tf in enumerate(time_fractions):
        cutoff = tf * planned_duration
        for j, bias in enumerate(bias_values):
            finish_opt  = _min_finish_biased(tasks, cutoff, bias, min_percentile)
            finish_base = _min_finish_biased(tasks, cutoff, bias, base_percentile)
            grid_opt[i, j]  = finish_opt  <= deadline
            grid_base[i, j] = finish_base <= deadline

    return grid_opt, grid_base, planned_duration, deadline


def compute_reachability_zones(
    task_file: str,
    time_fractions: np.ndarray,
    bias_values: np.ndarray,
    deadline: float | None = None,
    base_percentile: float = 0.80,
    min_percentile: float = 0.05,
    tolerance: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, float, float]:
    """
    Build a 2-D zone matrix for a fixed external deadline with tolerance band.

    Unlike ``compute_reachability_grid``, this function uses a single
    (optimistic) percentile to determine the earliest possible finish time
    for each cell, then classifies it into one of three zones based on its
    distance from the fixed deadline:

      - 2 (green):  ``min_finish ≤ deadline``
        Deadline is achievable.
      - 1 (yellow): ``deadline < min_finish ≤ deadline × (1 + tolerance)``
        Deadline is missed, but by no more than ``tolerance`` — still
        within the acceptable margin.
      - 0 (red):    ``min_finish > deadline × (1 + tolerance)``
        Deadline is clearly missed beyond the tolerance band.

    Args:
        task_file:       Path to the tasks CSV.
        time_fractions:  1-D array of project-time fractions in [0, 1].
        bias_values:     1-D array of bias factors.
        deadline:        Fixed external deadline in days.  If ``None``, the
                         planned project duration (from ``base_percentile``)
                         is used.
        base_percentile: Percentile for building the baseline plan.  Used
                         only to establish ``planned_duration`` (the axis
                         scale); the deadline may differ.
        min_percentile:  Most optimistic planning percentile for remaining
                         tasks (defines the "best case" finish time).
        tolerance:       Relative tolerance above the deadline that is still
                         considered acceptable (default 0.05 = ±5 %).
        seed:            RNG seed for the baseline ``build_schedule`` call.

    Returns:
        ``(zones, planned_duration, deadline_used)`` where ``zones`` is an
        integer matrix of shape ``(n_time, n_bias)`` with values 0, 1, or 2.
    """
    tasks = load_tasks_from_csv(task_file)
    build_schedule(tasks, percentile=base_percentile, seed=seed)
    planned_duration = max(t.planned_end_time for t in tasks)
    deadline_used = planned_duration if deadline is None else deadline
    hard_limit = deadline_used * (1.0 + tolerance)

    n_t = len(time_fractions)
    n_b = len(bias_values)
    zones = np.zeros((n_t, n_b), dtype=np.int8)

    for i, tf in enumerate(time_fractions):
        cutoff = tf * planned_duration
        for j, bias in enumerate(bias_values):
            min_finish = _min_finish_biased(tasks, cutoff, bias, min_percentile)
            if min_finish <= deadline_used:
                zones[i, j] = 2
            elif min_finish <= hard_limit:
                zones[i, j] = 1
            # else: 0 (red) — default

    return zones, planned_duration, deadline_used
