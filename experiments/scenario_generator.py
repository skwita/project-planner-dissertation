"""
Scenario generator for Bayesian replanning experiments.

Each scenario function takes the base task list and a seed, then simulates
what "actually happened" by drawing real durations under some perturbation
model.  The output is a ``progress`` dict ready to pass directly to
``reschedule_with_fixed_project_deadline``.

Scenario catalogue
------------------
1.  baseline            — draws from the true lognormal (no bias)
2.  high_variance       — same centre, 2x stddev (fatter tails)
3.  global_overrun      — systematic positive bias (e.g. 1.5x mean)
4.  global_underrun     — systematic negative bias (e.g. 0.67x mean)
5.  critical_task_late  — one task on the critical path is severely late
6.  late_discovery      — bias reveals itself only after the midpoint
7.  role_bottleneck     — one role (e.g. "разработчик") consistently over
8.  cascading_delay     — each task slightly worse than the previous
9.  early_wins_late_slip — first half fast, second half slow
10. perfect_execution   — everyone finishes exactly on the median
11. bimodal_outcomes    — tasks finish either very fast or very slow
12. deadline_impossible — bias so large that the deadline is unachievable
13. single_outlier      — one random non-critical task is 5x late
14. gradual_recovery    — starts with overrun, improves toward the end

All scenarios return the same structure:

    {
        task_id: {
            "status": "done" | "not_started",
            "actual_duration": float   # only for "done"
        },
        ...
    }

``n_observed`` controls how many tasks (in planned-end-time order) are
marked as "done"; the rest are "not_started".
"""

from __future__ import annotations

import numpy as np

from models.task import Task
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from utils.lognormal import lognorm_rvs, lognorm_log_params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw(mean: float, stddev: float, rng: np.random.Generator) -> float:
    """Draw one sample from LogNormal(mean, stddev) using the given RNG."""
    return lognorm_rvs(mean, stddev, rng)


def _build_progress(
    tasks: list[Task],
    actual_durations: dict[int, float],
    n_observed: int,
) -> dict[int, dict]:
    """
    Build a progress dict given actual durations and observation count.

    Tasks are sorted by planned_end_time; the first ``n_observed`` are
    marked "done", the rest "not_started".
    """
    ordered = sorted(
        tasks,
        key=lambda t: t.planned_end_time if t.planned_end_time is not None else 1e9,
    )
    progress: dict[int, dict] = {}
    for i, task in enumerate(ordered):
        if i < n_observed:
            progress[task.task_id] = {
                "status": "done",
                "actual_duration": actual_durations[task.task_id],
            }
        else:
            progress[task.task_id] = {"status": "not_started"}
    return progress


# ---------------------------------------------------------------------------
# Public API: scenario context manager
# ---------------------------------------------------------------------------

def load_scheduled_tasks(
    task_file: str = "data/tasks.csv",
    percentile: float = 0.9,
    seed: int = 42,
) -> list[Task]:
    """
    Load tasks and build a baseline scheduled plan.

    Returns the scheduled task list (with planned_* fields set).
    """
    tasks = load_tasks_from_csv(task_file)
    build_schedule(tasks, percentile=percentile, seed=seed)
    return tasks


# ---------------------------------------------------------------------------
# Scenario functions
# (all share the same signature for easy iteration in experiments)
# ---------------------------------------------------------------------------

ScenarioFn = "Callable[[list[Task], int, int], dict[int, dict]]"


def scenario_baseline(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
) -> dict[int, dict]:
    """
    Scenario 1 — Baseline: actuals drawn from the true lognormal.

    No bias, no extra variance.  The planner's model is correct.
    Expected result: replanner should not need to adjust much.
    """
    rng = np.random.default_rng(seed)
    actuals = {t.task_id: _draw(t.mean, t.stddev, rng) for t in tasks}
    return _build_progress(tasks, actuals, n_observed)


def scenario_high_variance(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    variance_multiplier: float = 2.0,
) -> dict[int, dict]:
    """
    Scenario 2 — High variance: same mean, stddev scaled by ``variance_multiplier``.

    Models a project where estimates are directionally correct but
    uncertainty is much higher than assumed (e.g. novel technology,
    unclear requirements).
    Expected result: wider posterior, replanner uses lower percentile.
    """
    rng = np.random.default_rng(seed)
    actuals = {
        t.task_id: _draw(t.mean, t.stddev * variance_multiplier, rng)
        for t in tasks
    }
    return _build_progress(tasks, actuals, n_observed)


def scenario_global_overrun(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    bias_factor: float = 1.5,
) -> dict[int, dict]:
    """
    Scenario 3 — Global overrun: all tasks take ``bias_factor`` × longer.

    Models optimism bias or scope creep that affects the whole team.
    Expected result: replanner detects positive θ and lowers percentile
    to keep the deadline.
    """
    rng = np.random.default_rng(seed)
    actuals = {
        t.task_id: _draw(t.mean * bias_factor, t.stddev * bias_factor, rng)
        for t in tasks
    }
    return _build_progress(tasks, actuals, n_observed)


def scenario_global_underrun(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    bias_factor: float = 0.67,
) -> dict[int, dict]:
    """
    Scenario 4 — Global underrun: all tasks finish faster than planned.

    Models a well-oiled team, or tasks that were conservatively estimated.
    Expected result: replanner detects negative θ and raises percentile
    (more comfortable plan with the saved time).
    """
    rng = np.random.default_rng(seed)
    actuals = {
        t.task_id: _draw(t.mean * bias_factor, t.stddev * bias_factor, rng)
        for t in tasks
    }
    return _build_progress(tasks, actuals, n_observed)


def scenario_critical_task_late(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    late_multiplier: float = 4.0,
) -> dict[int, dict]:
    """
    Scenario 5 — One critical task is severely late.

    The task with the highest mean duration (proxy for criticality) takes
    ``late_multiplier`` × its planned duration.  All other tasks are normal.
    Expected result: replanner detects overrun mainly on the critical path;
    may find deadline unachievable if the slip is large enough.
    """
    rng = np.random.default_rng(seed)
    # Pick the task with the highest planned duration as the "critical" one
    critical_task = max(tasks, key=lambda t: t.mean)
    actuals = {}
    for t in tasks:
        if t.task_id == critical_task.task_id:
            actuals[t.task_id] = t.mean * late_multiplier
        else:
            actuals[t.task_id] = _draw(t.mean, t.stddev, rng)
    return _build_progress(tasks, actuals, n_observed)


def scenario_late_discovery(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    bias_factor: float = 1.5,
    discovery_fraction: float = 0.5,
) -> dict[int, dict]:
    """
    Scenario 6 — Bias reveals itself only after the midpoint.

    The first ``discovery_fraction`` of tasks (by planned end time) run
    normally; the remaining ones run at ``bias_factor`` × mean.
    Models a situation where problems surface mid-project (e.g. hidden
    technical debt discovered during integration).
    Expected result: early replanning sees no signal; later replanning
    correctly adjusts.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(tasks, key=lambda t: t.planned_end_time or 1e9)
    cutoff = int(len(ordered) * discovery_fraction)
    normal_ids = {t.task_id for t in ordered[:cutoff]}

    actuals = {}
    for t in tasks:
        if t.task_id in normal_ids:
            actuals[t.task_id] = _draw(t.mean, t.stddev, rng)
        else:
            actuals[t.task_id] = _draw(
                t.mean * bias_factor, t.stddev * bias_factor, rng
            )
    return _build_progress(tasks, actuals, n_observed)


def scenario_role_bottleneck(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    bottleneck_role: str = "разработчик",
    bias_factor: float = 1.6,
) -> dict[int, dict]:
    """
    Scenario 7 — One role consistently overruns.

    Only tasks assigned to ``bottleneck_role`` take longer; other roles
    are on track.  Models a skill gap, understaffing, or technical issues
    affecting one team.
    Expected result: global bias estimate is diluted (signal-to-noise
    depends on fraction of tasks in that role).
    """
    rng = np.random.default_rng(seed)
    actuals = {}
    for t in tasks:
        if t.role == bottleneck_role:
            actuals[t.task_id] = _draw(
                t.mean * bias_factor, t.stddev * bias_factor, rng
            )
        else:
            actuals[t.task_id] = _draw(t.mean, t.stddev, rng)
    return _build_progress(tasks, actuals, n_observed)


def scenario_cascading_delay(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    start_bias: float = 1.0,
    end_bias: float = 2.0,
) -> dict[int, dict]:
    """
    Scenario 8 — Cascading delays: bias grows linearly over the project.

    The first task runs on time, the last task takes ``end_bias`` × mean.
    Models accumulating technical debt, morale degradation, or scope
    creep that compounds over time.
    Expected result: replanner underestimates bias if observed early;
    late-stage replanning sees a much stronger signal.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(tasks, key=lambda t: t.planned_end_time or 1e9)
    n = len(ordered)
    actuals = {}
    for i, t in enumerate(ordered):
        factor = start_bias + (end_bias - start_bias) * (i / max(n - 1, 1))
        actuals[t.task_id] = _draw(t.mean * factor, t.stddev * factor, rng)
    return _build_progress(tasks, actuals, n_observed)


def scenario_early_wins_late_slip(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    early_bias: float = 0.7,
    late_bias: float = 1.6,
    split_fraction: float = 0.5,
) -> dict[int, dict]:
    """
    Scenario 9 — Early wins, late slip.

    First half of tasks (by planned end time) run faster than planned;
    second half runs slower.  A realistic pattern when quick early wins
    give false confidence, masking problems in the harder back-end work.
    Expected result: replanner initially raises percentile (optimistic),
    then must lower it sharply once the slip appears.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(tasks, key=lambda t: t.planned_end_time or 1e9)
    cutoff = int(len(ordered) * split_fraction)
    early_ids = {t.task_id for t in ordered[:cutoff]}

    actuals = {}
    for t in tasks:
        factor = early_bias if t.task_id in early_ids else late_bias
        actuals[t.task_id] = _draw(t.mean * factor, t.stddev * factor, rng)
    return _build_progress(tasks, actuals, n_observed)


def scenario_perfect_execution(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
) -> dict[int, dict]:
    """
    Scenario 10 — Perfect execution: everyone finishes at exactly the median.

    Zero randomness — deterministic actuals equal to the 50th-percentile
    lognormal quantile.  A theoretical best-case for comparison.
    Expected result: replanner detects no bias; percentile stays at base.
    """
    from scipy.stats import lognorm as _lognorm
    actuals = {}
    for t in tasks:
        mu_log, sigma_log = lognorm_log_params(t.mean, t.stddev)
        actuals[t.task_id] = float(
            _lognorm.ppf(0.5, s=sigma_log, scale=np.exp(mu_log))
        )
    return _build_progress(tasks, actuals, n_observed)


def scenario_bimodal_outcomes(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    fast_factor: float = 0.5,
    slow_factor: float = 2.0,
    fast_probability: float = 0.4,
) -> dict[int, dict]:
    """
    Scenario 11 — Bimodal outcomes: tasks go either very fast or very slow.

    Each task independently finishes at ``fast_factor`` × mean (with
    probability ``fast_probability``) or ``slow_factor`` × mean otherwise.
    Models a context with high uncertainty and binary outcomes (e.g.
    research tasks that either "click" or require a complete redo).
    Expected result: high posterior variance; replanner is more conservative.
    """
    rng = np.random.default_rng(seed)
    actuals = {}
    for t in tasks:
        if rng.random() < fast_probability:
            actuals[t.task_id] = _draw(
                t.mean * fast_factor, t.stddev * fast_factor, rng
            )
        else:
            actuals[t.task_id] = _draw(
                t.mean * slow_factor, t.stddev * slow_factor, rng
            )
    return _build_progress(tasks, actuals, n_observed)


def scenario_deadline_impossible(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    bias_factor: float = 3.0,
) -> dict[int, dict]:
    """
    Scenario 12 — Deadline is unachievable.

    All tasks run at ``bias_factor`` × mean — so severe that no percentile
    choice can bring the project within the original deadline.
    Tests that the replanner correctly returns ``deadline_met=False`` and
    still produces the best possible plan.
    """
    rng = np.random.default_rng(seed)
    actuals = {
        t.task_id: _draw(t.mean * bias_factor, t.stddev, rng)
        for t in tasks
    }
    return _build_progress(tasks, actuals, n_observed)


def scenario_single_outlier(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    outlier_multiplier: float = 5.0,
) -> dict[int, dict]:
    """
    Scenario 13 — Single outlier: one random non-critical task is 5× late.

    Tests robustness of the bias estimator against a single extreme
    observation.  Because the Bayesian model treats all tasks as exchangeable
    (same global θ), one outlier should not dominate the posterior.
    """
    rng = np.random.default_rng(seed)
    # Choose a non-critical task: one with few or no dependents
    dependent_counts: dict[int, int] = {t.task_id: 0 for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            dependent_counts[dep] = dependent_counts.get(dep, 0) + 1
    # Pick the task with fewest dependents (least critical) among observed
    ordered = sorted(tasks, key=lambda t: t.planned_end_time or 1e9)
    observed_tasks = ordered[:n_observed]
    outlier = min(observed_tasks, key=lambda t: dependent_counts[t.task_id])

    actuals = {}
    for t in tasks:
        if t.task_id == outlier.task_id:
            actuals[t.task_id] = t.mean * outlier_multiplier
        else:
            actuals[t.task_id] = _draw(t.mean, t.stddev, rng)
    return _build_progress(tasks, actuals, n_observed)


def scenario_gradual_recovery(
    tasks: list[Task],
    n_observed: int,
    seed: int = 0,
    initial_bias: float = 1.8,
    final_bias: float = 1.0,
) -> dict[int, dict]:
    """
    Scenario 14 — Gradual recovery: starts overrunning, improves over time.

    The team adapts and recovers from initial difficulties.
    Bias decays linearly from ``initial_bias`` to ``final_bias``.
    Expected result: replanner should detect recovery signal and
    cautiously raise percentile as more data arrives.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(tasks, key=lambda t: t.planned_end_time or 1e9)
    n = len(ordered)
    actuals = {}
    for i, t in enumerate(ordered):
        factor = initial_bias + (final_bias - initial_bias) * (i / max(n - 1, 1))
        actuals[t.task_id] = _draw(t.mean * factor, t.stddev, rng)
    return _build_progress(tasks, actuals, n_observed)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "fn": scenario_baseline,
        "description": "True lognormal — no bias, no extra variance",
        "kwargs": {},
    },
    "high_variance": {
        "fn": scenario_high_variance,
        "description": "Same mean, 2× stddev (fatter tails)",
        "kwargs": {"variance_multiplier": 2.0},
    },
    "global_overrun_1.5x": {
        "fn": scenario_global_overrun,
        "description": "All tasks 1.5× longer (optimism bias)",
        "kwargs": {"bias_factor": 1.5},
    },
    "global_overrun_2x": {
        "fn": scenario_global_overrun,
        "description": "All tasks 2× longer (severe optimism bias)",
        "kwargs": {"bias_factor": 2.0},
    },
    "global_underrun": {
        "fn": scenario_global_underrun,
        "description": "All tasks 0.67× — conservative estimates",
        "kwargs": {"bias_factor": 0.67},
    },
    "critical_task_late": {
        "fn": scenario_critical_task_late,
        "description": "Longest task takes 4× planned",
        "kwargs": {"late_multiplier": 4.0},
    },
    "late_discovery": {
        "fn": scenario_late_discovery,
        "description": "Bias appears only after 50% completion",
        "kwargs": {"bias_factor": 1.5, "discovery_fraction": 0.5},
    },
    "role_bottleneck": {
        "fn": scenario_role_bottleneck,
        "description": "Developers take 1.6× longer; analysts/testers on track",
        "kwargs": {"bottleneck_role": "разработчик", "bias_factor": 1.6},
    },
    "cascading_delay": {
        "fn": scenario_cascading_delay,
        "description": "Bias grows from 1.0× to 2.0× over the project",
        "kwargs": {"start_bias": 1.0, "end_bias": 2.0},
    },
    "early_wins_late_slip": {
        "fn": scenario_early_wins_late_slip,
        "description": "First half 0.7×, second half 1.6×",
        "kwargs": {"early_bias": 0.7, "late_bias": 1.6},
    },
    "perfect_execution": {
        "fn": scenario_perfect_execution,
        "description": "Everyone finishes at exactly the median — zero noise",
        "kwargs": {},
    },
    "bimodal_outcomes": {
        "fn": scenario_bimodal_outcomes,
        "description": "40% chance fast (0.5×), 60% chance slow (2×)",
        "kwargs": {"fast_factor": 0.5, "slow_factor": 2.0, "fast_probability": 0.4},
    },
    "deadline_impossible": {
        "fn": scenario_deadline_impossible,
        "description": "All tasks 3× longer — deadline unachievable (stress test)",
        "kwargs": {"bias_factor": 3.0},
    },
    "single_outlier": {
        "fn": scenario_single_outlier,
        "description": "One non-critical task is 5× late; rest normal",
        "kwargs": {"outlier_multiplier": 5.0},
    },
    "gradual_recovery": {
        "fn": scenario_gradual_recovery,
        "description": "Starts at 1.8× bias, recovers to 1.0× by end",
        "kwargs": {"initial_bias": 1.8, "final_bias": 1.0},
    },
}
