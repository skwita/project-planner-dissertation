"""
Bayesian replanning engine.

Given a set of completed/in-progress tasks with their actual durations,
this module:

1. Estimates a global systematic bias in task duration estimates using a
   Bayesian Gaussian conjugate update (``estimate_global_bias_posterior``).
2. Reconstructs the factual schedule for completed tasks from time 0
   (``reconstruct_fact_schedule``).
3. Builds a new forward plan for remaining tasks using the updated
   predictive distributions (``build_rescheduled_plan``).
4. Searches for the *maximum* planning percentile that keeps the project
   within a fixed deadline (``reschedule_with_fixed_project_deadline``).

Progress dict format
--------------------
``progress`` maps task_id → status dict::

    {
        1: {"status": "done",         "actual_duration": 3.5},
        2: {"status": "in_progress",  "spent": 1.2},
        3: {"status": "not_started"},
    }
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm

from models.task import Task
from services.scheduler import topo_sort_tasks
from utils.lognormal import lognorm_log_params, lognorm_scipy_params, lognorm_ppf


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GlobalBiasPosterior:
    """Gaussian posterior over the log-scale systematic bias θ."""
    mean: float   # posterior mean for θ = ln(bias factor)
    var: float    # posterior variance


# ---------------------------------------------------------------------------
# Lognormal helpers
# ---------------------------------------------------------------------------

# lognorm helpers imported from utils.lognormal


# _percentile_duration replaced by lognorm_ppf from utils.lognormal


def _conditional_remaining(mu_log: float, sigma_log: float,
                            spent: float, percentile: float) -> float:
    """
    Conditional remaining duration for an in-progress task.

    Returns Q_p(T | T > spent) − spent, i.e. the expected remaining time
    given that the task has already consumed ``spent`` days.
    """
    dist = lognorm(s=sigma_log, scale=np.exp(mu_log))
    cdf_spent = float(dist.cdf(spent))

    if cdf_spent >= 0.999_999:
        return 0.0

    u = cdf_spent + percentile * (1.0 - cdf_spent)
    return max(0.0, float(dist.ppf(u)) - spent)


# ---------------------------------------------------------------------------
# Bayesian bias estimation
# ---------------------------------------------------------------------------

def estimate_global_bias_posterior(
    tasks: list[Task],
    progress: dict[int, dict],
    prior_mean: float = 0.0,
    prior_std: float = 0.30,
    obs_noise: float = 0.10,
) -> tuple[GlobalBiasPosterior, int]:
    """
    Estimate the global log-scale duration bias using Bayesian conjugate update.

    Model::

        ln(T_actual_i) = mu_log_i + θ + ε_i
        θ ~ N(prior_mean, prior_std²)
        ε_i ~ N(0, sigma_log_i² + obs_noise²)

    Args:
        tasks:       All project tasks (only completed ones are used).
        progress:    Status dict (see module docstring).
        prior_mean:  Prior mean for θ (0 = no expected bias).
        prior_std:   Prior std for θ (larger = more uncertain about bias).
        obs_noise:   Additional observation noise in log-space.

    Returns:
        Tuple of (posterior, n_observations_used).
    """
    prior_var = prior_std ** 2
    precision = 1.0 / prior_var
    weighted_sum = prior_mean / prior_var
    used = 0

    for task in tasks:
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] != "done":
            continue
        actual = float(info["actual_duration"])
        if actual <= 0:
            continue

        mu_log, sigma_log = lognorm_log_params(task.mean, task.stddev)
        z = np.log(actual) - mu_log
        obs_var = sigma_log ** 2 + obs_noise ** 2

        precision += 1.0 / obs_var
        weighted_sum += z / obs_var
        used += 1

    post_var = 1.0 / precision
    post_mean = post_var * weighted_sum
    return GlobalBiasPosterior(mean=post_mean, var=post_var), used


# ---------------------------------------------------------------------------
# Predictive distribution for a single future task
# ---------------------------------------------------------------------------

def _predictive_params(task: Task,
                        posterior: GlobalBiasPosterior) -> tuple[float, float]:
    """
    Shift the task's lognormal distribution by the estimated bias.

    Returns (mu_pred, sigma_pred) for a lognormal that accounts for both
    the task's own uncertainty and the posterior uncertainty about θ.
    """
    mu_log, sigma_log = lognorm_log_params(task.mean, task.stddev)
    mu_pred = mu_log + posterior.mean
    sigma_pred = float(np.sqrt(sigma_log ** 2 + posterior.var))
    return mu_pred, sigma_pred


# ---------------------------------------------------------------------------
# Factual schedule reconstruction
# ---------------------------------------------------------------------------

def reconstruct_fact_schedule(
    tasks: list[Task],
    progress: dict[int, dict],
) -> tuple[list[Task], float]:
    """
    Reconstruct the actual (historical) schedule for completed tasks.

    Completed tasks are placed in dependency + resource order starting from
    time 0, using their reported ``actual_duration``.  Incomplete tasks are
    left with ``real_*`` fields as ``None``.

    Returns:
        (tasks_copy, current_time) where ``current_time`` is the latest
        real end time among completed tasks.
    """
    tasks = deepcopy(tasks)
    task_map = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    current_time = 0.0
    role_real_ready: dict[str, float] = defaultdict(float)

    for task_id in order:
        task = task_map[task_id]
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] != "done":
            continue

        dep_end = max(
            (task_map[dep].real_end_time
             for dep in task.dependencies
             if task_map[dep].real_end_time is not None),
            default=0.0,
        )
        duration = float(info["actual_duration"])

        # Mirror plan-driven execution: a task cannot start before its
        # originally planned date (team follows the schedule even when
        # predecessors finish early), and a role cannot work two tasks
        # simultaneously.  Without these floors, reconstructed timings
        # are artificially compressed, making baseline appear to finish
        # far ahead of the deadline.
        planned_start = task.planned_start_time if task.planned_start_time is not None else 0.0
        task.real_start_time = max(dep_end, planned_start, role_real_ready[task.role])
        task.real_duration = duration
        task.real_end_time = task.real_start_time + duration
        role_real_ready[task.role] = task.real_end_time
        current_time = max(current_time, task.real_end_time)

    return tasks, current_time


# ---------------------------------------------------------------------------
# Forward replanning
# ---------------------------------------------------------------------------

def build_rescheduled_plan(
    tasks: list[Task],
    progress: dict[int, dict],
    current_time: float,
    percentile: float,
    posterior: GlobalBiasPosterior,
) -> tuple[list[Task], float, float]:
    """
    Build a new forward plan using the bias-adjusted predictive distributions.

    Processing order:
      1. **Done** tasks: planned times mirror reconstructed actual times.
      2. **In-progress** tasks: start from ``current_time``; remaining
         duration is the conditional quantile given time already spent.
      3. **Not-started** tasks: scheduled fresh using the predictive quantile.

    Args:
        tasks:        All project tasks (will be deep-copied internally).
        progress:     Status dict.
        current_time: Earliest possible start for not-started tasks.
        percentile:   Planning quantile for remaining / new tasks.
        posterior:    Bias posterior from ``estimate_global_bias_posterior``.

    Returns:
        Tuple of (replanned_tasks, project_finish, current_time).
    """
    fact_tasks, reconstructed_time = reconstruct_fact_schedule(tasks, progress)
    current_time = max(current_time, reconstructed_time)

    tasks = deepcopy(fact_tasks)
    task_map = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    role_ready: dict[str, float] = defaultdict(float)

    # Pass 1: fix completed tasks
    for task in tasks:
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] == "done":
            task.planned_start_time = task.real_start_time
            task.planned_duration = task.real_duration
            task.planned_end_time = task.real_end_time
            role_ready[task.role] = max(role_ready[task.role], task.planned_end_time)

    # Pass 2: in-progress tasks (conditional remaining duration)
    for task_id in order:
        task = task_map[task_id]
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] != "in_progress":
            continue

        dep_end = max(
            (task_map[dep].planned_end_time for dep in task.dependencies),
            default=0.0,
        )
        start = max(dep_end, role_ready[task.role])
        spent = float(info["spent"])

        mu_pred, sigma_pred = _predictive_params(task, posterior)
        remaining = _conditional_remaining(mu_pred, sigma_pred, spent, percentile)

        task.real_start_time = start
        task.real_duration = spent
        task.real_end_time = start + spent
        task.planned_start_time = start
        task.planned_duration = spent + remaining
        task.planned_end_time = start + spent + remaining

        role_ready[task.role] = task.planned_end_time
        current_time = max(current_time, task.real_end_time)

    # Pass 3: not-started tasks
    done_ids = {
        tid for tid, info in progress.items()
        if info.get("status") in ("done", "in_progress")
    }
    for task_id in order:
        task = task_map[task_id]
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] in ("done", "in_progress"):
            continue

        dep_end = max(
            (task_map[dep].planned_end_time for dep in task.dependencies),
            default=0.0,
        )

        # Apply current_time floor only if this task is genuinely
        # sequentially blocked: it has at least one predecessor that is
        # already done/in-progress (so it was waiting in real time).
        # Tasks on independent parallel branches keep their plan timing.
        has_done_predecessor = any(dep in done_ids for dep in task.dependencies)
        effective_floor = current_time if has_done_predecessor else 0.0

        start = max(dep_end, role_ready[task.role], effective_floor)

        mu_pred, sigma_pred = _predictive_params(task, posterior)
        duration = float(lognorm.ppf(percentile, s=sigma_pred, scale=np.exp(mu_pred)))

        task.planned_start_time = start
        task.planned_duration = duration
        task.planned_end_time = start + duration
        role_ready[task.role] = task.planned_end_time

    project_finish = max(
        (t.planned_end_time for t in tasks if t.planned_end_time is not None),
        default=0.0,
    )
    return tasks, project_finish, current_time


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reschedule_with_fixed_project_deadline(
    tasks: list[Task],
    progress: dict[int, dict],
    target_finish_time: float,
    current_time: float | None = None,
    base_percentile: float = 0.80,
    prev_percentile: float | None = None,
    prior_mean: float = 0.0,
    prior_std: float = 0.30,
    obs_noise: float = 0.10,
    min_percentile: float = 0.05,
    max_percentile: float = 0.99,
    max_iter: int = 50,
    tolerance: float = 1e-4,
) -> dict:
    """
    Replan the project targeting ``base_percentile``, adjusting only when
    the estimated bias forces a change to meet the deadline.

    Algorithm
    ---------
    1. Estimate the global bias posterior from completed tasks.
    2. Reconstruct the factual schedule to establish ``current_time``.
    3. Try ``base_percentile`` first (bias-adjusted).
       - If it meets the deadline → return it unchanged.  When there is no
         bias the plan stays exactly as originally conceived.
       - If it overshoots → binary-search *downward* from ``base_percentile``
         to ``min_percentile`` for the highest p that still meets the deadline.
         This happens when a positive bias is detected and the project is
         genuinely at risk.
       - If it undershoots significantly (bias is negative, project ahead of
         schedule) → binary-search *upward* from ``base_percentile`` to
         ``max_percentile`` to reclaim the slack as a more conservative buffer.

    The key invariant: **without detected bias the returned percentile equals
    ``base_percentile`` and the plan is effectively unchanged.**

    Args:
        tasks:               All project tasks.
        progress:            Status dict.
        target_finish_time:  Hard deadline (same unit as task durations).
        current_time:        Override for the reconstructed current time
                             (``None`` = derive from completed tasks).
        base_percentile:     Target planning percentile.  The algorithm
                             tries to stay at this value and only deviates
                             when the bias estimate requires it.
        prev_percentile:     Percentile returned by the previous replanning
                             call.  When all tasks are done (nothing left to
                             schedule), this value is echoed back unchanged
                             instead of defaulting to ``base_percentile``,
                             keeping convergence traces smooth.  ``None``
                             falls back to ``base_percentile``.
        prior_mean:          Prior mean for log-bias θ (0 = no expected bias).
        prior_std:           Prior std for θ.
        obs_noise:           Log-space observation noise.
        min_percentile:      Hard lower bound (used when deadline is at risk).
        max_percentile:      Hard upper bound (used when reclaiming slack).
        max_iter:            Binary-search iteration limit.
        tolerance:           Convergence threshold in percentile units.

    Returns:
        Dict with keys:
          - ``tasks``             — Replanned Task list.
          - ``project_finish``    — Expected finish time of the new plan.
          - ``current_time``      — Reconstructed current time.
          - ``used_percentile``   — The percentile that was applied.
          - ``deadline_met``      — Whether the deadline is achievable.
          - ``posterior``         — The ``GlobalBiasPosterior`` object.
          - ``used_observations`` — Number of completed tasks used.
          - ``bias_factor_mean``  — exp(posterior.mean).
          - ``message``           — Human-readable status string.
    """
    posterior, used = estimate_global_bias_posterior(
        tasks, progress, prior_mean, prior_std, obs_noise
    )

    _, reconstructed_time = reconstruct_fact_schedule(tasks, progress)
    if current_time is None:
        current_time = reconstructed_time

    def _base_result(p: float, t_tasks: list[Task],
                     finish: float, met: bool, msg: str) -> dict:
        return {
            "tasks": t_tasks,
            "project_finish": finish,
            "current_time": current_time,
            "used_percentile": p,
            "deadline_met": met,
            "posterior": posterior,
            "used_observations": used,
            "bias_factor_mean": float(np.exp(posterior.mean)),
            "message": msg,
        }

    def _plan(p: float) -> tuple[list[Task], float]:
        t, f, _ = build_rescheduled_plan(tasks, progress, current_time, p, posterior)
        return t, f

    # ------------------------------------------------------------------ #
    # Step 1: try base_percentile as-is                                  #
    # ------------------------------------------------------------------ #
    base_tasks, base_finish = _plan(base_percentile)

    # When all tasks are done, project_finish == current_time (Pass 3 is empty).
    # No remaining work to schedule → any percentile gives the same finish.
    # Skip the binary search entirely and carry forward prev_percentile so the
    # convergence trace doesn't show an artificial spike to max_percentile.
    if abs(base_finish - current_time) < 1e-9:
        fallback_p = prev_percentile if prev_percentile is not None else base_percentile
        return _base_result(
            fallback_p, base_tasks, base_finish,
            base_finish <= target_finish_time,
            "All tasks complete; reporting factual finish.",
        )

    if base_finish <= target_finish_time:
        # Deadline met at base_percentile — no need to change anything.
        # Only search upward if bias is meaningfully negative (project
        # is genuinely ahead of schedule), so we can reclaim the slack
        # as a more conservative buffer.  Threshold: posterior.mean < -0.05
        # (bias factor < ~0.95) to avoid chasing noise.
        if posterior.mean < -0.05:
            # Search upward: find highest p ∈ [base_percentile, max_percentile]
            # that still fits.
            hi_tasks, hi_finish = _plan(max_percentile)
            if hi_finish <= target_finish_time:
                return _base_result(
                    max_percentile, hi_tasks, hi_finish, True,
                    "Negative bias detected; raised percentile to reclaim slack.",
                )
            lo, hi = base_percentile, max_percentile
            best_tasks, best_finish, best_p = base_tasks, base_finish, base_percentile
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                t, f = _plan(mid)
                if f <= target_finish_time:
                    best_tasks, best_finish, best_p = t, f, mid
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tolerance:
                    break
            return _base_result(
                best_p, best_tasks, best_finish, True,
                f"Negative bias (factor {np.exp(posterior.mean):.3f}); "
                f"raised percentile from {base_percentile:.3f} to {best_p:.3f}.",
            )

        # No significant bias — return base plan unchanged.
        return _base_result(
            base_percentile, base_tasks, base_finish, True,
            "No significant bias detected; plan unchanged.",
        )

    # ------------------------------------------------------------------ #
    # Step 2: base_percentile overshoots — deadline is at risk            #
    # Positive bias detected.  Search downward to find the highest p      #
    # in [min_percentile, base_percentile] that still meets the deadline. #
    # ------------------------------------------------------------------ #
    min_tasks, min_finish = _plan(min_percentile)
    if min_finish > target_finish_time:
        return _base_result(
            min_percentile, min_tasks, min_finish, False,
            "Deadline is unachievable even at the minimum percentile.",
        )

    lo, hi = min_percentile, base_percentile
    best_tasks, best_finish, best_p = min_tasks, min_finish, min_percentile

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t, f = _plan(mid)
        if f <= target_finish_time:
            best_tasks, best_finish, best_p = t, f, mid
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance:
            break

    return _base_result(
        best_p, best_tasks, best_finish, True,
        f"Positive bias (factor {np.exp(posterior.mean):.3f}); "
        f"lowered percentile from {base_percentile:.3f} to {best_p:.3f} "
        f"to protect the deadline.",
    )
