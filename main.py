"""
Dissertation project planner — main analysis script.

Runs the full pipeline:
  Part 1.1  Build an averaged Monte Carlo schedule → Gantt + Excel export
  Part 1.2  Explore the effect of the task percentile on project duration
  Part 1.3  Estimate the project buffer at a given confidence level
  Part 1.4  Pareto plot: idle time vs. project duration
  Part 1.5  Overlay PDF/CDF for a selected set of percentiles
  Part 1.6  Heatmaps of duration, idle time and buffer

Each function is self-contained and can be called individually.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from concurrent import futures
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from services.bayes_rescheduler import (
    build_rescheduled_plan,
    estimate_global_bias_posterior,
    reconstruct_fact_schedule,
    reschedule_with_fixed_project_deadline,
)
from services.critical_path import critical_chain_path
from services.exporter import export_percentile_analysis_to_excel, export_schedule_to_excel
from services.metrics import (
    calculate_buffer,
    calculate_idle_time,
    calculate_project_duration,
    monte_carlo_simulation,
    parallel_monte_carlo_simulation,
)
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from visualization.gantt_chart import plot_gantt, plot_replanned_gantt
from visualization.plot_idle_vs_duration import (
    plot_idle_vs_duration,
    plot_history_metrics,
    plot_pareto_transition,
    plot_pareto_shift_trajectory,
)
from visualization.plot_percentiles_ends_distr import plot_percentile_cdfs, plot_percentile_pdf

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(label: str) -> None:
    """Print a section separator with a timestamp."""
    sep = "_" * 54
    print(f"\n{sep}\n{label} started at {datetime.now().time()}\n{sep}")


# ---------------------------------------------------------------------------
# Part 1.1 — Schedule project and export
# ---------------------------------------------------------------------------

def part1_1_schedule_project(
    pr_buffer: float,
    path: str = "data/tasks.csv",
    percentile: float = 0.9,
    export_excel: bool = True,
    runs: int = 10_000,
):
    """
    Build a schedule by averaging ``runs`` Monte Carlo simulations.

    The averaged times are used for the Gantt chart and, optionally, the
    Excel export. This approach smooths out per-run randomness.

    Args:
        pr_buffer:    Project buffer in days (drawn on the Gantt).
        path:         Path to the tasks CSV.
        percentile:   Task planning percentile.
        export_excel: Whether to write ``output/output_schedule.xlsx``.
        runs:         Number of MC runs to average.

    Returns:
        Tuple of (scheduled_tasks, project_duration, idle_time).
    """
    _log("Part 1.1")

    tasks = load_tasks_from_csv(path)
    aggregated: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run in range(runs):
        tasks_copy = copy.deepcopy(tasks)
        scheduled = build_schedule(tasks_copy, percentile, seed=run)
        for task in scheduled:
            aggregated[task.task_id]["planned_start_time"].append(task.planned_start_time)
            aggregated[task.task_id]["planned_end_time"].append(task.planned_end_time)
            aggregated[task.task_id]["real_start_time"].append(task.real_start_time)
            aggregated[task.task_id]["real_end_time"].append(task.real_end_time)

    # Replace individual run values with run averages
    scheduled_tasks = copy.deepcopy(tasks)
    for task in scheduled_tasks:
        agg = aggregated[task.task_id]
        task.planned_start_time = float(np.mean(agg["planned_start_time"]))
        task.planned_end_time   = float(np.mean(agg["planned_end_time"]))
        task.real_start_time    = float(np.mean(agg["real_start_time"]))
        task.real_end_time      = float(np.mean(agg["real_end_time"]))
        task.planned_duration   = task.planned_end_time - task.planned_start_time
        task.real_duration      = task.real_end_time - task.real_start_time

    project_duration = calculate_project_duration(scheduled_tasks)
    idle = calculate_idle_time(scheduled_tasks)

    plot_gantt(scheduled_tasks, f"output/plots/gantt_{percentile}.png", pr_buffer)

    if export_excel:
        export_schedule_to_excel(
            scheduled_tasks,
            filename="output/output_schedule.xlsx",
            project_duration=project_duration,
            idle_time=idle,
        )

    return scheduled_tasks, project_duration, idle


# ---------------------------------------------------------------------------
# Part 1.2 — Explore percentile effect
# ---------------------------------------------------------------------------

def part1_2_explore_percentile_effect(
    percentiles: list[float],
    task_file: str = "data/tasks.csv",
    n_iter: int = 100_000,
    seed: int | None = None,
):
    """
    Run MC simulations across multiple planning percentiles and export results.

    Saves individual PDF and CDF plots per percentile, plus an Excel summary.

    Args:
        percentiles: List of task percentiles to evaluate.
        task_file:   Path to tasks CSV.
        n_iter:      MC iterations per percentile.
        seed:        RNG seed.

    Returns:
        DataFrame with the summary statistics.
    """
    _log("Part 1.2")

    parallel_results = parallel_monte_carlo_simulation(task_file, percentiles, n_iter, seed)
    results = []
    all_durations: dict[float, np.ndarray] = {}

    for p in percentiles:
        durations, idles, _ = parallel_results[p]
        all_durations[p] = durations

        roles = {role for idle in idles for role in idle}
        avg_idle = {role: np.mean([idle.get(role, 0.0) for idle in idles]) for role in roles}

        row: dict = {"Percentile": p, "Mean project duration": round(float(np.mean(durations)), 2)}
        for role, val in avg_idle.items():
            row[f"Idle_{role}"] = round(val, 2)
        results.append(row)

    all_values = np.concatenate(list(all_durations.values()))
    xmin, xmax = int(all_values.min() - 1), int(all_values.max() + 1)

    for p, durations in all_durations.items():
        plot_percentile_pdf(
            [durations], [f"p={p:.2f}"],
            filename=f"output/plots/pdf_percentile_{p:.2f}.png",
            bins=xmax - xmin, xlim=(xmin, xmax),
        )
        plot_percentile_cdfs(
            [durations], [f"p={p:.2f}"],
            filename=f"output/plots/cdf_percentile_{p:.2f}.png",
            xlim=(xmin, xmax),
        )

    return export_percentile_analysis_to_excel(results, "output/percentile_analysis.xlsx")


# ---------------------------------------------------------------------------
# Part 1.3 — Project buffer
# ---------------------------------------------------------------------------

def part1_3_project_buffer(
    percentile_tasks: float = 0.5,
    percentile_project: float = 0.9,
    task_file: str = "data/tasks.csv",
    n_iter: int = 1_000,
    seed: int | None = None,
) -> float:
    """
    Estimate the project buffer at the given confidence level.

    The buffer is defined as the ``percentile_project``-th percentile of
    schedule overruns relative to the planned project duration.

    Args:
        percentile_tasks:   Task planning percentile.
        percentile_project: Confidence level for the buffer (e.g. 90 = 90th %).
        task_file:          Path to tasks CSV.
        n_iter:             MC iterations.
        seed:               RNG seed.

    Returns:
        Buffer size in days.
    """
    tasks = load_tasks_from_csv(task_file)
    scheduled_tasks = build_schedule(tasks, percentile=percentile_tasks, seed=seed)
    planned_duration = calculate_project_duration(scheduled_tasks)

    durations, _, _ = monte_carlo_simulation(task_file, percentile_tasks, n_iter, seed, 0)
    return calculate_buffer(durations, planned_duration, percentile_project)


# ---------------------------------------------------------------------------
# Part 1.4 — Pareto idle vs duration
# ---------------------------------------------------------------------------

def compute_pareto_idle_duration_curve(
    percentiles_tasks: list[float],
    task_file: str = "data/tasks.csv",
    seed: int | None = None,
    n_iter: int = 100_000,
) -> tuple[list[float], list[float]]:
    """
    Compute the raw data behind the duration/idle Pareto curve.

    For each task percentile, runs a Monte Carlo simulation and averages
    the resulting project duration and total per-role idle time. This is
    the same data ``part1_4_plot_pareto_idle_vs_duration`` plots; it is
    exposed separately so other callers (e.g. the Pareto-shift trajectory
    plot) can reuse it as the "baseline plan" curve.

    Args:
        percentiles_tasks: List of task percentiles.
        task_file:         Path to tasks CSV.
        seed:              RNG seed.
        n_iter:            MC iterations per percentile.

    Returns:
        Tuple of (mean_durations, mean_total_idle) — parallel lists, one
        entry per percentile in ``percentiles_tasks``.
    """
    parallel_results = parallel_monte_carlo_simulation(task_file, percentiles_tasks, n_iter, seed)
    durations, idles_sum = [], []

    for p in percentiles_tasks:
        mc_durations, mc_idles, _ = parallel_results[p]
        durations.append(float(np.mean(mc_durations)))
        idles_sum.append(float(np.mean([sum(idle.values()) for idle in mc_idles])))

    return durations, idles_sum


def part1_4_plot_pareto_idle_vs_duration(
    percentiles_tasks: list[float],
    task_file: str = "data/tasks.csv",
    seed: int | None = None,
    n_iter: int = 100_000,
    save_path: str = "output/plots/pareto_idle_duration.png",
) -> None:
    """
    Build and save the Pareto scatter plot: duration vs. idle time.

    Args:
        percentiles_tasks: List of task percentiles.
        task_file:         Path to tasks CSV.
        seed:              RNG seed.
        n_iter:            MC iterations per percentile.
        save_path:         Output image path.
    """
    _log("Part 1.4")

    durations, idles_sum = compute_pareto_idle_duration_curve(
        percentiles_tasks, task_file, seed, n_iter
    )

    plot_idle_vs_duration(durations, idles_sum, percentiles_tasks, n_iter, save_path)


# ---------------------------------------------------------------------------
# Part 1.5 — Overlay multiple percentiles
# ---------------------------------------------------------------------------

def part1_5_multiple_percentiles(
    percentiles: list[float],
    task_file: str = "data/tasks.csv",
    seed: int | None = None,
) -> None:
    """
    Overlay PDF and CDF curves for a selection of percentiles.

    Args:
        percentiles: Percentiles to overlay (e.g. [0.1, 0.5, 0.9]).
        task_file:   Path to tasks CSV.
        seed:        RNG seed.
    """
    _log("Part 1.5")

    parallel_results = parallel_monte_carlo_simulation(task_file, percentiles, 1000, seed)
    all_durations = [parallel_results[p][0] for p in percentiles]

    plot_percentile_pdf(
        all_durations, percentiles,
        "output/plots/project_duration_distributions_multiple_percentiles_pdf.png",
    )
    plot_percentile_cdfs(
        all_durations, percentiles,
        "output/plots/project_duration_distributions_multiple_percentiles_cdf.png",
    )


# ---------------------------------------------------------------------------
# Part 1.6 — Heatmaps
# ---------------------------------------------------------------------------

def part1_6_plot_heatmaps(
    task_percentiles: list[float],
    project_percentiles: list[float],
) -> None:
    """Run all three heatmap analyses."""
    _log("Part 1.6")
    _heatmap_durations(task_percentiles, project_percentiles)
    _heatmap_idles(task_percentiles, project_percentiles)
    _heatmap_project_buffer(task_percentiles, project_percentiles)


# ---- Helper workers (must be module-level for multiprocessing) ----

def _compute_duration_and_buffer(task_file: str, t_p: float, p_p: float,
                                  n_sim: int, seed: int | None) -> tuple[float, float]:
    durations, _, _ = monte_carlo_simulation(task_file, t_p, n_sim, seed, 0)
    buffer = part1_3_project_buffer(t_p, p_p * 100)
    return float(np.mean(durations)), buffer


def _compute_avg_idle(task_file: str, t_p: float,
                      n_sim: int, seed: int | None) -> float:
    _, idles, _ = monte_carlo_simulation(task_file, t_p, n_sim, seed, 0)
    return float(np.mean([sum(idle.values()) for idle in idles]))


def _save_heatmap(matrix: np.ndarray, task_percentiles: list[float],
                  project_percentiles: list[float], title: str, path: str) -> None:
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="mako",
        xticklabels=[f"p={p:.2f}" for p in project_percentiles],
        yticklabels=[f"p={t:.2f}" for t in task_percentiles],
    )
    ax.invert_yaxis()
    plt.xlabel("Project percentile")
    plt.ylabel("Task percentile")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _heatmap_durations(
    task_file: str = "data/tasks.csv",
    task_percentiles: list[float] | None = None,
    project_percentiles: list[float] | None = None,
    seed: int | None = None,
    n_sim: int = 100,
) -> None:
    task_percentiles = task_percentiles or [0.5, 0.7, 0.9]
    project_percentiles = project_percentiles or [0.5, 0.7, 0.9]
    matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    with futures.ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(_compute_duration_and_buffer, task_file, t_p, p_p, n_sim, seed): (i, j)
            for i, t_p in enumerate(task_percentiles)
            for j, p_p in enumerate(project_percentiles)
        }
        for future in tqdm(futures.as_completed(future_to_idx),
                           total=len(future_to_idx), desc="Duration heatmap"):
            i, j = future_to_idx[future]
            avg_dur, buffer = future.result()
            matrix[i, j] = avg_dur + buffer

    _save_heatmap(matrix, task_percentiles, project_percentiles,
                  "Heatmap: project duration (with buffer)",
                  "output/plots/heatmap_durations_with_buffer.png")


def _heatmap_idles(
    task_file: str = "data/tasks.csv",
    task_percentiles: list[float] | None = None,
    project_percentiles: list[float] | None = None,
    seed: int | None = None,
    n_sim: int = 100,
) -> None:
    task_percentiles = task_percentiles or [0.5, 0.7, 0.9]
    project_percentiles = project_percentiles or [0.5, 0.7, 0.9]
    matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    with futures.ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(_compute_avg_idle, task_file, t_p, n_sim, seed): (i, j)
            for i, t_p in enumerate(task_percentiles)
            for j, _ in enumerate(project_percentiles)
        }
        for future in tqdm(futures.as_completed(future_to_idx),
                           total=len(future_to_idx), desc="Idle heatmap"):
            i, j = future_to_idx[future]
            matrix[i, j] = future.result()

    _save_heatmap(matrix, task_percentiles, project_percentiles,
                  "Heatmap: idle time",
                  "output/plots/heatmap_idle.png")


def _heatmap_project_buffer(
    task_file: str = "data/tasks.csv",
    task_percentiles: list[float] | None = None,
    project_percentiles: list[float] | None = None,
    seed: int | None = None,
    n_sim: int = 100,
) -> None:
    task_percentiles = task_percentiles or [0.5, 0.7, 0.9]
    project_percentiles = project_percentiles or [0.5, 0.7, 0.9]
    matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    with futures.ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(part1_3_project_buffer, t_p, p_p * 100): (i, j)
            for i, t_p in enumerate(task_percentiles)
            for j, p_p in enumerate(project_percentiles)
        }
        for future in tqdm(futures.as_completed(future_to_idx),
                           total=len(future_to_idx), desc="Buffer heatmap"):
            i, j = future_to_idx[future]
            try:
                matrix[i, j] = future.result()
            except Exception as exc:
                matrix[i, j] = np.nan
                print(f"  ⚠ Buffer calc failed at ({i}, {j}): {exc}")

    _save_heatmap(matrix, task_percentiles, project_percentiles,
                  "Heatmap: project buffer",
                  "output/plots/heatmap_buffer.png")


# ---------------------------------------------------------------------------
# Part 3 — Bayesian replanning (single snapshot)
# ---------------------------------------------------------------------------

def part3_bayesian_replanning(
    task_file: str = "data/tasks.csv",
    base_percentile: float = 0.80,
    progress: dict | None = None,
    prior_mean: float = 0.0,
    prior_std: float = 0.30,
    obs_noise: float = 0.10,
    export_plot: bool = True,
) -> dict:
    """
    Replan the project from the current observed state.

    Uses completed task actuals to estimate a global duration bias via
    Bayesian conjugate update, then finds the highest planning percentile
    that keeps the replanned finish within the original deadline.

    Args:
        task_file:       Path to tasks CSV.
        base_percentile: Percentile used to build the original baseline plan.
        progress:        Status dict {task_id: {"status": ..., ...}}.
                         Defaults to empty (no observations yet).
        prior_mean:      Prior mean for log-bias θ (0 = no expected bias).
        prior_std:       Prior std for log-bias θ.
        obs_noise:       Log-space observation noise.
        export_plot:     If True, saves a replanned Gantt chart.

    Returns:
        Result dict from ``reschedule_with_fixed_project_deadline``.
    """
    _log("Part 3 — Bayesian replanning")

    if progress is None:
        progress = {}

    # Baseline plan to establish the original deadline
    original_tasks = load_tasks_from_csv(task_file)
    build_schedule(original_tasks, percentile=base_percentile, seed=42)
    target_finish_time = max(t.planned_end_time for t in original_tasks)

    tasks_for_replanning = load_tasks_from_csv(task_file)
    result = reschedule_with_fixed_project_deadline(
        tasks=tasks_for_replanning,
        progress=progress,
        target_finish_time=target_finish_time,
        current_time=None,
        base_percentile=base_percentile,
        prior_mean=prior_mean,
        prior_std=prior_std,
        obs_noise=obs_noise,
    )

    print(f"  Observations used:    {result['used_observations']}")
    print(f"  Bias factor (exp θ):  {result['bias_factor_mean']:.4f}")
    print(f"  Chosen percentile:    {result['used_percentile']:.4f}")
    print(f"  Replanned finish:     {result['project_finish']:.2f}")
    print(f"  Deadline met:         {result['deadline_met']}")
    print(f"  Message:              {result['message']}")

    if export_plot:
        plot_replanned_gantt(
            original_tasks=original_tasks,
            replanned_tasks=result["tasks"],
            filename="output/plots/replanned_gantt.png",
            target_finish_time=target_finish_time,
        )

    return result


# ---------------------------------------------------------------------------
# Part 4 — Iterative Bayesian replanning
# ---------------------------------------------------------------------------

def part4_multistage_replanning_iterative(
    full_progress: dict,
    task_file: str = "data/tasks.csv",
    base_percentile: float = 0.8,
    batch_size: int = 5,
    prior_mean: float = 0.0,
    prior_std: float = 0.3,
    obs_noise: float = 0.1,
    export_gantts: bool = False,
) -> list[dict]:
    """
    Simulate multi-stage replanning as actuals arrive batch by batch.

    Tasks are revealed in order of their planned end time.  At each stage,
    ``batch_size`` additional completed tasks are added to the observed
    progress and the plan is updated.  The result history can be visualised
    with ``plot_history_metrics``.

    Args:
        full_progress:   Complete actual progress dict (all tasks).
        task_file:       Path to tasks CSV.
        base_percentile: Percentile for the initial baseline plan.
        batch_size:      Number of tasks revealed per iteration.
        prior_mean:      Prior mean for log-bias θ.
        prior_std:       Prior std for log-bias θ.
        obs_noise:       Log-space observation noise.
        export_gantts:   If True, saves a Gantt chart after each stage.

    Returns:
        List of stage summary dicts with keys: ``stage``, ``completed``,
        ``bias``, ``percentile``, ``finish``.
    """
    _log("Part 4 — Iterative replanning")

    # Initial baseline plan
    current_tasks = load_tasks_from_csv(task_file)
    build_schedule(current_tasks, percentile=base_percentile, seed=42)
    target_finish_time = max(t.planned_end_time for t in current_tasks)

    ordered_ids = [
        t.task_id for t in sorted(current_tasks, key=lambda x: x.planned_end_time)
    ]

    observed_progress: dict = {}
    history: list[dict] = []

    for stage_start in range(0, len(ordered_ids), batch_size):
        stage_num = stage_start // batch_size + 1
        batch_ids = ordered_ids[stage_start: stage_start + batch_size]

        # Reveal next batch of actuals
        for tid in batch_ids:
            if tid in full_progress:
                observed_progress[tid] = full_progress[tid]

        print(f"\n=== Stage {stage_num} | observed: {sorted(observed_progress)} ===")

        result = reschedule_with_fixed_project_deadline(
            tasks=current_tasks,   # iterative: pass updated tasks each time
            progress=observed_progress,
            target_finish_time=target_finish_time,
            current_time=None,
            base_percentile=base_percentile,
            prior_mean=prior_mean,
            prior_std=prior_std,
            obs_noise=obs_noise,
        )
        current_tasks = result["tasks"]   # carry forward the updated plan

        print(f"  Bias:       {result['bias_factor_mean']:.3f}")
        print(f"  Percentile: {result['used_percentile']:.3f}")
        print(f"  Finish:     {result['project_finish']:.2f}")

        history.append({
            "stage":      stage_num,
            "completed":  len(observed_progress),
            "bias":       result["bias_factor_mean"],
            "percentile": result["used_percentile"],
            "finish":     result["project_finish"],
        })

        if export_gantts:
            plot_replanned_gantt(
                original_tasks=current_tasks,
                replanned_tasks=current_tasks,
                filename=f"output/plots/replanned_stage_{stage_num}.png",
                target_finish_time=target_finish_time,
            )

    return history


# ---------------------------------------------------------------------------
# Part 5 — Pareto front shift trajectory (deadline drift ↔ percentile compensation)
# ---------------------------------------------------------------------------

def build_pareto_shift_updates(
    full_progress: dict,
    task_file: str = "data/tasks.csv",
    base_percentile: float = 0.8,
    n_updates: int = 5,
    prior_mean: float = 0.0,
    prior_std: float = 0.3,
    obs_noise: float = 0.1,
) -> list[dict]:
    """
    Build the (drift, compensate) point pairs behind the Pareto-shift chart.

    Tasks are revealed cumulatively in ``n_updates`` equal shares (default 5
    → every 20 % of tasks, ordered by planned end time). At each update,
    two points are computed against the *same* refreshed bias posterior:

      1. **Drift** — the expected finish time at the percentile *currently
         governing the plan* (the previous update's compensate result, or
         ``base_percentile`` for the very first update) — unchanged, only
         the bias estimate is refreshed. As newly observed actuals refine
         the duration bias, this point moves horizontally away from the
         previous one (same percentile ⇒ same place on the Pareto front,
         only the deadline shifts) — this is the "deadline shift caused by
         refined estimates".
      2. **Compensate** — the planning percentile is searched (same
         algorithm as ``reschedule_with_fixed_project_deadline``, anchored
         at ``base_percentile``) to pull the finish time back toward the
         original deadline. Because percentile is exactly the parameter
         that traces the duration/idle Pareto front, this move is
         diagonal: part of the deadline drift is undone, at the cost (or
         benefit) of changed resource idle time. The resulting percentile
         becomes the new "currently governing" percentile for the next
         update's drift point.

    Note: this function only produces the ``finish`` (duration) coordinate
    of each point, since that is what the bias/percentile search actually
    computes. The ``effort`` (idle-time) coordinate is a property of *which
    percentile* was used, not of this deterministic single replanning —
    ``part5_pareto_shift_trajectory`` fills it in by mapping each point's
    percentile onto the baseline duration/idle Pareto curve.

    Args:
        full_progress:   Complete actual progress dict (all tasks).
        task_file:       Path to tasks CSV.
        base_percentile: Percentile for the initial baseline plan and for
                         each update's drift point.
        n_updates:       Number of equally-spaced updates (default 5 = every
                         20 % of tasks revealed).
        prior_mean:      Prior mean for log-bias θ.
        prior_std:       Prior std for log-bias θ.
        obs_noise:       Log-space observation noise.

    Returns:
        List of point dicts, starting with one ``"baseline"`` point followed
        by ``n_updates`` ``("drift", "compensate")`` pairs. Each dict has
        keys ``kind``, ``update`` (0 for baseline, else 1..n_updates),
        ``finish``, ``percentile``, and ``bias`` (exp of the posterior mean).
    """
    tasks = load_tasks_from_csv(task_file)
    build_schedule(tasks, percentile=base_percentile, seed=42)
    target_finish_time = max(t.planned_end_time for t in tasks)

    ordered_ids = [
        t.task_id for t in sorted(tasks, key=lambda x: x.planned_end_time)
    ]
    n_tasks = len(ordered_ids)

    points: list[dict] = [{
        "kind": "baseline",
        "update": 0,
        "finish": target_finish_time,
        "percentile": base_percentile,
        "bias": 1.0,
    }]

    current_tasks = copy.deepcopy(tasks)
    observed_progress: dict = {}
    prev_percentile = base_percentile

    for k in range(1, n_updates + 1):
        cutoff = round(n_tasks * k / n_updates)
        for tid in ordered_ids[:cutoff]:
            if tid in full_progress:
                observed_progress[tid] = full_progress[tid]

        print(f"\n=== Update {k}/{n_updates} | observed: {sorted(observed_progress)} ===")

        posterior, used = estimate_global_bias_posterior(
            current_tasks, observed_progress, prior_mean, prior_std, obs_noise,
        )
        _, current_time = reconstruct_fact_schedule(current_tasks, observed_progress)

        # --- 1) Drift point: percentile currently governing the plan
        #        (prev_percentile — the previous update's compensate result,
        #        or base_percentile for the very first update), refreshed
        #        bias. Same percentile as the point we're moving from ⇒
        #        same height on the baseline curve ⇒ a horizontal move. ---
        _, drift_finish, _ = build_rescheduled_plan(
            current_tasks, observed_progress, current_time, prev_percentile, posterior,
        )
        print(f"  Drift:      finish={drift_finish:.2f}  bias={np.exp(posterior.mean):.3f}")
        points.append({
            "kind": "drift",
            "update": k,
            "finish": drift_finish,
            "percentile": prev_percentile,
            "bias": float(np.exp(posterior.mean)),
        })

        # --- 2) Compensate point: search percentile to restore deadline ---
        result = reschedule_with_fixed_project_deadline(
            tasks=current_tasks,
            progress=observed_progress,
            target_finish_time=target_finish_time,
            current_time=current_time,
            base_percentile=base_percentile,
            prev_percentile=prev_percentile,
            prior_mean=prior_mean,
            prior_std=prior_std,
            obs_noise=obs_noise,
        )
        print(f"  Compensate: finish={result['project_finish']:.2f}  "
              f"percentile={result['used_percentile']:.3f}")
        points.append({
            "kind": "compensate",
            "update": k,
            "finish": result["project_finish"],
            "percentile": result["used_percentile"],
            "bias": result["bias_factor_mean"],
        })

        current_tasks = result["tasks"]
        prev_percentile = result["used_percentile"]

    return points


def part5_pareto_shift_trajectory(
    full_progress: dict,
    task_file: str = "data/tasks.csv",
    base_percentile: float = 0.8,
    n_updates: int = 5,
    baseline_percentiles: list[float] | None = None,
    baseline_n_iter: int = 20_000,
    prior_mean: float = 0.0,
    prior_std: float = 0.3,
    obs_noise: float = 0.1,
    seed: int | None = None,
    save_path: str = "output/plots/pareto_shift_trajectory.png",
) -> list[dict]:
    """
    Plot the Pareto-front shift: deadline drift vs. percentile compensation.

    Two things are drawn on the duration/idle-time plane:
      - The **baseline** curve — the original Pareto front implied by a
        percentile sweep at project start (same data as Part 1.4): "where
        the plan said the project would be."
      - The **update trajectory** — for each of ``n_updates`` equally-spaced
        data reveals (see ``build_pareto_shift_updates``), a horizontal
        drift move (deadline shift from refined bias, percentile held
        fixed) followed by a diagonal compensate move (percentile search
        pulling the plan back toward the deadline): "where the project
        actually is." Each point's idle-time (effort) coordinate is read
        off the baseline curve at that point's percentile — so a drift
        point (same percentile as before) keeps the same height, and a
        compensate point (new percentile) lands back on the curve.

    Args:
        full_progress:         Complete actual progress dict (see
                              ``reschedule_with_fixed_project_deadline``).
        task_file:              Path to tasks CSV.
        base_percentile:        Percentile for the initial baseline plan.
        n_updates:               Number of equally-spaced updates (default 5
                              = every 20 % of tasks revealed).
        baseline_percentiles:  Percentiles swept for the baseline curve.
                              Defaults to 0.05..0.95 in steps of 0.05.
        baseline_n_iter:        MC iterations per percentile for the baseline curve.
        prior_mean:             Prior mean for log-bias θ.
        prior_std:              Prior std for log-bias θ.
        obs_noise:              Log-space observation noise.
        seed:                   RNG seed for the baseline curve simulation.
        save_path:              Output image path.

    Returns:
        The point list from ``build_pareto_shift_updates``.
    """
    _log("Part 5 — Pareto shift trajectory")

    baseline_percentiles = baseline_percentiles or list(np.arange(0.05, 0.96, 0.05).tolist())
    baseline_durations, baseline_idles = compute_pareto_idle_duration_curve(
        baseline_percentiles, task_file, seed, baseline_n_iter,
    )

    points = build_pareto_shift_updates(
        full_progress=full_progress,
        task_file=task_file,
        base_percentile=base_percentile,
        n_updates=n_updates,
        prior_mean=prior_mean,
        prior_std=prior_std,
        obs_noise=obs_noise,
    )

    # Effort (idle-time) is a property of *which percentile* a point uses,
    # not of a single deterministic replan — read it off the baseline
    # curve so drift points (unchanged percentile) sit at a fixed height
    # and compensate points (new percentile) land back on the curve.
    pct_arr = np.asarray(baseline_percentiles, dtype=float)
    idle_arr = np.asarray(baseline_idles, dtype=float)
    order = np.argsort(pct_arr)
    for pt in points:
        pt["effort"] = float(np.interp(pt["percentile"], pct_arr[order], idle_arr[order]))

    original_tasks = load_tasks_from_csv(task_file)
    build_schedule(original_tasks, percentile=base_percentile, seed=42)
    deadline = max(t.planned_end_time for t in original_tasks)

    plot_pareto_shift_trajectory(
        baseline_durations=baseline_durations,
        baseline_idles=baseline_idles,
        points=points,
        deadline=deadline,
        save_path=save_path,
    )
    return points


if __name__ == "__main__":
    PERCENTILE_TASK = 0.74
    PERCENTILE_PROJECT = 0.9
    PERCENTILES_RANGE = list(np.arange(0.05, 0.96, 0.05).tolist())
    PERCENTILES_FOR_PLOT = [0.1, 0.5, 0.9]

    _log("Main")

    # ------------------------------------------------------------------ #
    # Part 1 — Monte Carlo scheduling analysis                           #
    # ------------------------------------------------------------------ #
    # pr_buffer = part1_3_project_buffer(
    #     percentile_tasks=PERCENTILE_TASK,
    #     percentile_project=PERCENTILE_PROJECT * 100,
    # )
    # part1_1_schedule_project(pr_buffer, percentile=PERCENTILE_TASK)
    # part1_2_explore_percentile_effect(percentiles=PERCENTILES_RANGE)
    # part1_4_plot_pareto_idle_vs_duration(PERCENTILES_RANGE)
    part1_5_multiple_percentiles([0.74])
    # part1_6_plot_heatmaps(
    #     task_percentiles=PERCENTILES_RANGE,
    #     project_percentiles=PERCENTILES_RANGE,
    # )

    # ------------------------------------------------------------------ #
    # Progress scenarios for Parts 3 & 4                                 #
    # Tasks marked "not_started" will be replanned from scratch.         #
    # ------------------------------------------------------------------ #

    # Scenario A: tasks run ~50 % over budget (bias ≈ 1.5×)
    progress_long = {
        1:  {"status": "done", "actual_duration": 3.0},
        2:  {"status": "done", "actual_duration": 9.0},
        3:  {"status": "done", "actual_duration": 12.0},
        4:  {"status": "done", "actual_duration": 6.0},
        5:  {"status": "done", "actual_duration": 7.5},
        6:  {"status": "done", "actual_duration": 5.25},
        7:  {"status": "done", "actual_duration": 15.0},
        8:  {"status": "done", "actual_duration": 18.0},
        9:  {"status": "done", "actual_duration": 12.0},
        10: {"status": "done", "actual_duration": 9.0},
        11: {"status": "done", "actual_duration": 7.5},
        12: {"status": "done", "actual_duration": 9.0},
        19: {"status": "done", "actual_duration": 9.0},
        20: {"status": "done", "actual_duration": 15.0},
        21: {"status": "done", "actual_duration": 12.0},
        **{tid: {"status": "not_started"} for tid in range(13, 46)
           if tid not in (19, 20, 21)},
    }

    # # Scenario B: tasks run ~33 % under budget (bias ≈ 0.67×)
    # progress_short = {
    #     1:  {"status": "done", "actual_duration": 1.33},
    #     2:  {"status": "done", "actual_duration": 4.0},
    #     3:  {"status": "done", "actual_duration": 5.33},
    #     4:  {"status": "done", "actual_duration": 2.67},
    #     5:  {"status": "done", "actual_duration": 3.33},
    #     6:  {"status": "done", "actual_duration": 2.33},
    #     7:  {"status": "done", "actual_duration": 6.67},
    #     8:  {"status": "done", "actual_duration": 8.0},
    #     9:  {"status": "done", "actual_duration": 5.33},
    #     10: {"status": "done", "actual_duration": 4.0},
    #     11: {"status": "done", "actual_duration": 3.33},
    #     12: {"status": "done", "actual_duration": 4.0},
    #     19: {"status": "done", "actual_duration": 4.0},
    #     20: {"status": "done", "actual_duration": 6.67},
    #     21: {"status": "done", "actual_duration": 5.33},
    #     **{tid: {"status": "not_started"} for tid in range(13, 46)
    #        if tid not in (19, 20, 21)},
    # }

    # # Scenario C: mixed wave — first half over, second half under budget
    # progress_wave = {tid: {"status": "done", "actual_duration": dur}
    #                  for tid, dur in [
    #     (1, 3.0), (2, 9.0), (3, 12.0), (4, 6.0), (5, 7.5),
    #     (6, 5.25), (7, 15.0), (8, 8.0), (9, 5.33), (10, 4.0),
    #     (11, 3.33), (12, 4.0), (13, 4.67), (14, 5.33),
    #     (15, 15.0), (16, 18.0), (17, 12.0), (18, 6.0),
    #     (19, 9.0), (20, 15.0), (21, 12.0), (22, 6.67),
    #     (23, 4.0), (24, 6.67), (25, 5.33), (26, 8.0),
    #     (27, 6.67), (28, 4.0), (29, 10.5), (30, 15.0),
    #     (31, 12.0), (32, 12.0), (33, 10.5), (34, 15.0),
    #     (35, 7.5), (36, 4.0), (37, 4.67), (38, 4.0),
    #     (39, 3.33), (40, 2.0), (41, 6.67), (42, 9.33),
    #     (43, 7.5), (44, 10.5), (45, 6.0),
    # ]}

    # # Scenario D: same wave but second half also over budget
    # progress_wave_worse = {tid: {"status": "done", "actual_duration": dur}
    #                        for tid, dur in [
    #     (1, 3.0), (2, 9.0), (3, 12.0), (4, 6.0), (5, 7.5),
    #     (6, 5.25), (7, 15.0),
    #     (8, 14.4), (9, 9.6), (10, 7.2), (11, 6.0), (12, 7.2),
    #     (13, 8.4), (14, 9.6),
    #     (15, 15.0), (16, 18.0), (17, 12.0), (18, 6.0),
    #     (19, 9.0), (20, 15.0), (21, 12.0),
    #     (22, 12.0), (23, 7.2), (24, 12.0), (25, 9.6),
    #     (26, 14.4), (27, 12.0), (28, 7.2),
    #     (29, 10.5), (30, 15.0), (31, 12.0), (32, 12.0),
    #     (33, 10.5), (34, 15.0), (35, 7.5),
    #     (36, 7.2), (37, 8.4), (38, 7.2), (39, 6.0), (40, 3.6),
    #     (41, 12.0), (42, 16.8),
    #     (43, 7.5), (44, 10.5), (45, 6.0),
    # ]}

    # # ------------------------------------------------------------------ #
    # # Part 3 — Single-snapshot Bayesian replanning                       #
    # # ------------------------------------------------------------------ #
    # result = part3_bayesian_replanning(
    #     task_file="data/tasks.csv",
    #     base_percentile=PERCENTILE_TASK,
    #     progress=progress_long,
    #     prior_mean=0.0,
    #     prior_std=0.30,
    #     obs_noise=0.10,
    #     export_plot=True,
    # )

    # # ------------------------------------------------------------------ #
    # # Part 4 — Iterative replanning across stages                        #
    # # ------------------------------------------------------------------ #
    # history = part4_multistage_replanning_iterative(
    #     full_progress=progress_wave_worse,
    #     task_file="data/tasks.csv",
    #     base_percentile=PERCENTILE_TASK,
    #     batch_size=2,
    #     prior_mean=0.0,
    #     prior_std=0.30,
    #     obs_noise=0.10,
    #     export_gantts=True,
    # )
    # plot_history_metrics(history)

    # ------------------------------------------------------------------ #
    # Part 5 — Pareto front shift trajectory: scenario comparison         #
    # ------------------------------------------------------------------ #
    _pareto_shift_base_tasks = load_tasks_from_csv("data/tasks.csv")

    def _uniform_bias_progress(factor: float) -> dict:
        """Every task overruns (or underruns) by the same relative factor."""
        return {
            t.task_id: {"status": "done", "actual_duration": round(t.mean * factor, 3)}
            for t in _pareto_shift_base_tasks
        }

    pareto_shift_scenarios = {
        # Uniform 20 % overrun (bias ≈ 1.2×) — spreads the 5 updates out
        # instead of saturating min_percentile after the first update.
        "uniform_1_20x": _uniform_bias_progress(1.20),
        # Uniform 20 % underrun (bias ≈ 0.8×) — mirror case: the project
        # runs ahead of schedule, so compensation should reclaim slack by
        # *raising* the percentile instead of lowering it.
        "uniform_0_80x": _uniform_bias_progress(0.80),
        # Mild uniform 5 % overrun (bias ≈ 1.05×) — small enough that the
        # deadline stays reachable throughout without hitting min_percentile.
        "uniform_1_05x": _uniform_bias_progress(1.05),
        # Everything on-estimate except task 7, which blows out ~5x — shows
        # how a single critical-task outlier pulls the *global* bias
        # estimate (shared across all tasks) once it's revealed.
        "critical_task_7_overrun": {
            t.task_id: {
                "status": "done",
                "actual_duration": t.mean if t.task_id != 7 else round(t.mean * 5, 3),
            }
            for t in _pareto_shift_base_tasks
        },
    }

    for _scenario_name, _scenario_progress in pareto_shift_scenarios.items():
        part5_pareto_shift_trajectory(
            full_progress=_scenario_progress,
            task_file="data/tasks.csv",
            base_percentile=PERCENTILE_TASK,
            n_updates=5,
            prior_mean=0.0,
            prior_std=0.30,
            obs_noise=0.10,
            save_path=f"output/plots/pareto_shift_trajectory_{_scenario_name}.png",
        )
