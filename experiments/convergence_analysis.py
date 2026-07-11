"""
Convergence analysis for the Bayesian replanner.

Core question: *does it make sense to replan after every single task?*

Short answer — yes, with caveats:
  - The posterior update is cheap (O(n) in the number of observations).
  - Early updates carry little information; the bias estimate has high
    variance until ~20–30 % of tasks are observed.
  - After that, each new observation shrinks the posterior and the chosen
    percentile stabilises.  The convergence plots make this visible.

What this module produces
-------------------------
For each scenario × seed combination, ``run_convergence`` replans after
every single completed task and records the three key metrics:

    bias_factor   — exp(posterior.mean), the multiplicative duration bias
    percentile    — the planning quantile chosen by the binary search
    finish        — expected project finish time under the new plan

``plot_convergence_grid`` then draws one figure per scenario with:
  - Three vertically stacked subplots (bias / percentile / finish)
  - Multiple light seed-lines in the background (showing run variance)
  - A bold mean line in the foreground
  - Reference lines: true bias, base percentile, and the deadline
  - A shaded "low-information zone" (first 20 % of tasks) where the
    posterior has not yet converged

Usage
-----
    python -m experiments.convergence_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.scenario_generator import SCENARIOS, load_scheduled_tasks
from services.bayes_rescheduler import (
    reconstruct_fact_schedule,
    reschedule_with_fixed_project_deadline,
)
from services.metrics import calculate_idle_time
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from services.reachability import compute_reachability_grid, compute_reachability_zones
from visualization.plot_reachability import (
    plot_deadline_reachability,
    plot_reachability_with_tolerance,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_FILE        = "data/tasks.csv"
BASE_PERCENTILE  = 0.9
PLAN_SEED        = 42
N_SEEDS          = 8           # seed-lines per scenario plot
OUTPUT_DIR       = Path("output/experiments/convergence")

# Scenarios to include in the convergence analysis.
# Subset chosen to show diverse convergence behaviours.
CONVERGENCE_SCENARIOS = [
    "baseline",
    "global_overrun_1.5x",
    "global_overrun_2x",
    "global_underrun",
    "high_variance",
    "cascading_delay",
    "early_wins_late_slip",
    "late_discovery",
    "role_bottleneck",
    "critical_task_late",
    "bimodal_outcomes",
    "gradual_recovery",
    "single_outlier",
    "deadline_impossible",
]

# True bias factors for each scenario (for the reference line on the bias plot)
TRUE_BIAS: dict[str, float] = {
    "baseline":           1.00,
    "global_overrun_1.5x": 1.50,
    "global_overrun_2x":  2.00,
    "global_underrun":    0.67,
    "high_variance":      1.00,
    "cascading_delay":    1.50,   # geometric mean of 1.0→2.0
    "early_wins_late_slip": 1.15, # rough average of 0.7 and 1.6
    "late_discovery":     1.25,   # average of normal + 1.5×
    "role_bottleneck":    1.00,   # diluted (only developers affected)
    "critical_task_late": 1.00,   # only one task; global bias ≈ 1
    "bimodal_outcomes":   1.30,   # 0.4×0.5 + 0.6×2.0 in expectation
    "gradual_recovery":   1.40,   # average of 1.8→1.0
    "single_outlier":     1.00,   # one task; should be robust
    "deadline_impossible": 3.00,
    "perfect_execution":  1.00,
}


# ---------------------------------------------------------------------------
# Single convergence trace
# ---------------------------------------------------------------------------

def run_convergence(
    scenario_name: str,
    seed: int,
    scheduled_tasks,
    target_finish_time: float,
    task_file: str = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
    baseline_tasks=None,
) -> pd.DataFrame:
    """
    Replan after *every* completed task and record metrics at each step.

    Returns a DataFrame with columns:
        n_completed   — number of tasks marked done (1 … N)
        bias_factor   — exp(posterior.mean)
        percentile    — chosen planning percentile
        finish        — replanned project finish time
        deadline_met  — bool
        posterior_std — sqrt(posterior.var), measure of remaining uncertainty
        idle_time     — cumulative cross-role idle time among done tasks (days)

    Args:
        baseline_tasks: Original scheduled task list used as the reference
                        plan for idle-time comparison.  When None, falls
                        back to ``scheduled_tasks``.
    """
    if baseline_tasks is None:
        baseline_tasks = scheduled_tasks
    scenario   = SCENARIOS[scenario_name]
    n_tasks    = len(scheduled_tasks)

    # Generate ALL actuals for this scenario up front
    full_progress: dict[int, dict] = scenario["fn"](
        scheduled_tasks,
        n_observed=n_tasks,   # reveal everything
        seed=seed,
        **scenario["kwargs"],
    )

    # Ordered task IDs by planned_end_time (observation order)
    ordered_ids = [
        t.task_id
        for t in sorted(scheduled_tasks,
                        key=lambda t: t.planned_end_time or 1e9)
    ]

    records = []
    observed: dict[int, dict] = {}
    last_percentile: float | None = None

    for step, tid in enumerate(ordered_ids, start=1):
        # Reveal one more task
        observed[tid] = full_progress[tid]

        # Fresh tasks with baseline plan for the rescheduler
        fresh = load_tasks_from_csv(task_file)
        build_schedule(fresh, percentile=base_percentile, seed=PLAN_SEED)

        try:
            result = reschedule_with_fixed_project_deadline(
                tasks=fresh,
                progress=observed,
                target_finish_time=target_finish_time,
                base_percentile=base_percentile,
                prev_percentile=last_percentile,
            )
            last_percentile = result["used_percentile"]

            # Idle time: compare actual done-task starts vs. original plan
            fact_tasks, _ = reconstruct_fact_schedule(fresh, observed)
            idle_by_role   = calculate_idle_time(fact_tasks, baseline_tasks)
            idle_total     = sum(idle_by_role.values())

            records.append({
                "n_completed":   step,
                "pct_completed": step / n_tasks,
                "bias_factor":   result["bias_factor_mean"],
                "percentile":    result["used_percentile"],
                "finish":        result["project_finish"],
                "deadline_met":  result["deadline_met"],
                "posterior_std": float(np.sqrt(result["posterior"].var)),
                "idle_time":     idle_total,
            })
        except Exception as exc:
            records.append({
                "n_completed":   step,
                "pct_completed": step / n_tasks,
                "bias_factor":   np.nan,
                "percentile":    np.nan,
                "finish":        np.nan,
                "deadline_met":  False,
                "posterior_std": np.nan,
                "idle_time":     np.nan,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Run all convergence traces
# ---------------------------------------------------------------------------

def run_all_convergence(
    scenarios: list[str] = CONVERGENCE_SCENARIOS,
    n_seeds: int = N_SEEDS,
    task_file: str = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
    plan_seed: int = PLAN_SEED,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> dict[str, list[pd.DataFrame]]:
    """
    Run convergence traces for all requested scenarios.

    Returns:
        Dict mapping scenario_name → list of DataFrames (one per seed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduled_tasks = load_scheduled_tasks(task_file, base_percentile, plan_seed)
    n_tasks = len(scheduled_tasks)
    target_finish = max(t.planned_end_time for t in scheduled_tasks)

    if verbose:
        total = len(scenarios) * n_seeds * n_tasks
        print(f"Baseline deadline: {target_finish:.1f} days | "
              f"{n_tasks} tasks | {len(scenarios)} scenarios | "
              f"{n_seeds} seeds")
        print(f"Total replanner calls: {total}\n")

    results: dict[str, list[pd.DataFrame]] = {}

    for s_idx, scenario_name in enumerate(scenarios, 1):
        if scenario_name not in SCENARIOS:
            print(f"  [SKIP] Unknown scenario: {scenario_name}")
            continue

        traces = []
        for seed in range(n_seeds):
            if verbose:
                print(f"  [{s_idx}/{len(scenarios)}] {scenario_name:<30} seed={seed}")
            df = run_convergence(
                scenario_name=scenario_name,
                seed=seed,
                scheduled_tasks=scheduled_tasks,
                target_finish_time=target_finish,
                task_file=task_file,
                base_percentile=base_percentile,
                baseline_tasks=scheduled_tasks,
            )
            traces.append(df)
        results[scenario_name] = traces

    return results, target_finish, n_tasks


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

_METRIC_META = [
    # (column,        ylabel,                  title)
    ("bias_factor",  "Bias factor  exp(θ)",    "Bias estimate convergence"),
    ("percentile",   "Planning percentile",     "Percentile convergence"),
    ("finish",       "Project finish (days)",   "Finish time convergence"),
    ("idle_time",    "Cross-role idle (days)",  "Labor costs (idle time)"),
]


def plot_convergence_grid(
    results: dict[str, list[pd.DataFrame]],
    target_finish: float,
    n_tasks: int,
    base_percentile: float = BASE_PERCENTILE,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> None:
    """
    One figure per scenario: 3 stacked subplots (bias / percentile / finish).

    Visual elements:
      - Light grey lines: individual seed traces (show run-to-run variance)
      - Bold coloured line: mean across seeds
      - Dashed reference lines: true bias, base percentile, deadline
      - Red shaded region: first 20 % of tasks ("low-information zone")
      - Posterior std band: ±1 std of bias estimate shaded around mean
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name, traces in results.items():
        if not traces:
            continue

        true_bias     = TRUE_BIAS.get(scenario_name, 1.0)
        desc          = SCENARIOS[scenario_name]["description"]

        # Align all traces on n_completed (they should all be identical length)
        x = traces[0]["pct_completed"].values * 100   # 0–100 %

        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        fig.suptitle(
            f"Convergence: {scenario_name}\n{desc}",
            fontsize=11, fontweight="bold",
        )

        refs   = [true_bias,      base_percentile, target_finish, None]
        labels = ["true bias",    "base percentile", "deadline",   None]
        colors = ["forestgreen",  "steelblue",       "red",        None]

        for ax, (col, ylabel, title), ref, ref_label, ref_color in zip(
            axes, _METRIC_META, refs, labels, colors
        ):
            # --- Individual seed traces (background) ---
            seed_matrix = np.array([tr[col].values for tr in traces])

            for row in seed_matrix:
                ax.plot(x, row, color="silver", linewidth=0.8, alpha=0.7)

            # --- Mean line ---
            mean_line = np.nanmean(seed_matrix, axis=0)
            ax.plot(x, mean_line, color="black", linewidth=2.0,
                    label="mean across seeds")

            # --- ±1 std band (only for bias — most informative there) ---
            if col == "bias_factor":
                std_line = np.nanstd(seed_matrix, axis=0)
                ax.fill_between(x,
                                mean_line - std_line,
                                mean_line + std_line,
                                alpha=0.15, color="black",
                                label="±1 std (seeds)")

            # --- Posterior uncertainty band from first trace ---
            if col == "bias_factor" and "posterior_std" in traces[0].columns:
                post_std = np.nanmean(
                    np.array([tr["posterior_std"].values for tr in traces]),
                    axis=0,
                )
                ax.fill_between(x,
                                mean_line - post_std,
                                mean_line + post_std,
                                alpha=0.10, color="steelblue",
                                label="±posterior std (mean)")

            # --- Reference line (skip when None) ---
            if ref is not None:
                ax.axhline(ref, linestyle="--", linewidth=1.4,
                           color=ref_color, alpha=0.8, label=ref_label)

            # --- Low-information zone (first 20 %) ---
            ax.axvspan(0, 20, alpha=0.06, color="red",
                       label="low-information zone" if ax is axes[0] else None)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=9, pad=3)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=7, loc="upper right", ncol=2)

        axes[-1].set_xlabel("Tasks completed (%)", fontsize=10)

        # --- Deadline-met rate annotation on finish subplot (index 2) ---
        deadline_rates = []
        pct_breakpoints = [25, 50, 75]
        for pct in pct_breakpoints:
            step = max(1, round(pct / 100 * n_tasks)) - 1
            rates = [
                tr["deadline_met"].iloc[step]
                if step < len(tr) else np.nan
                for tr in traces
            ]
            deadline_rates.append(np.nanmean(rates))
        annotation = "  |  ".join(
            f"{p}%→{r:.0%}" for p, r in zip(pct_breakpoints, deadline_rates)
        )
        axes[2].set_title(
            f"Finish time convergence   [deadline met — {annotation}]",
            fontsize=9, pad=3,
        )

        plt.tight_layout()
        safe = scenario_name.replace("/", "_").replace(".", "_")
        path = output_dir / f"convergence_{safe}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"    → {path}")


# ---------------------------------------------------------------------------
# Cross-scenario summary plot
# ---------------------------------------------------------------------------

def plot_convergence_summary(
    results: dict[str, list[pd.DataFrame]],
    target_finish: float,
    n_tasks: int,
    base_percentile: float = BASE_PERCENTILE,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    One figure comparing all scenarios on a single set of axes.

    Three panels, each showing the mean convergence curve per scenario.
    Useful for a dissertation overview figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("Convergence summary — all scenarios", fontsize=12,
                 fontweight="bold")

    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(results), 1)) for i in range(len(results))]

    refs       = [1.0,            base_percentile, target_finish, None]
    ylabels    = ["Bias factor exp(θ)", "Planning percentile",
                  "Project finish (days)", "Cross-role idle (days)"]
    titles     = ["Bias estimate", "Percentile", "Finish time",
                  "Labor costs (idle time)"]
    ref_labels = ["no bias", "base percentile", "deadline", None]

    for ax, col, ylabel, title, ref, ref_label in zip(
        axes,
        ["bias_factor", "percentile", "finish", "idle_time"],
        ylabels, titles, refs, ref_labels,
    ):
        for (scenario_name, traces), color in zip(results.items(), colors):
            if not traces:
                continue
            x = traces[0]["pct_completed"].values * 100
            seed_matrix = np.array([tr[col].values for tr in traces])
            mean_line   = np.nanmean(seed_matrix, axis=0)
            ax.plot(x, mean_line, linewidth=1.6, color=color,
                    label=scenario_name, alpha=0.85)

        if ref is not None:
            ax.axhline(ref, linestyle="--", linewidth=1.2,
                       color="black", alpha=0.5, label=ref_label)
        ax.axvspan(0, 20, alpha=0.05, color="red")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=3)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Tasks completed (%)", fontsize=10)

    # Shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = output_dir / "convergence_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary plot → {path}")


# ---------------------------------------------------------------------------
# "Does it make sense?" — quantitative answer
# ---------------------------------------------------------------------------

def compute_convergence_stats(
    results: dict[str, list[pd.DataFrame]],
    n_tasks: int,
    target_finish: float,
) -> pd.DataFrame:
    """
    For each scenario, compute:
      - ``convergence_step``: first step where bias estimate stabilises
        (posterior_std drops below 0.05 — a practical convergence threshold)
      - ``convergence_pct``: corresponding fraction of tasks completed
      - ``bias_rmse_early``: RMSE of bias estimate in the first 20 % of tasks
      - ``bias_rmse_late``: RMSE of bias estimate in the last 30 % of tasks
      - ``deadline_met_at_50pct``: fraction of seeds meeting deadline
        when 50 % of tasks are done
      - ``percentile_range``: max − min chosen percentile across the run
        (large → replanner had to adapt a lot)

    This table directly answers "does replanning after every task help?"
    — specifically, at what point does the additional information stop
    being useful.
    """
    rows = []
    cutoff_early = int(0.20 * n_tasks)
    cutoff_late  = int(0.70 * n_tasks)
    step_50pct   = int(0.50 * n_tasks) - 1

    for scenario_name, traces in results.items():
        if not traces:
            continue

        true_bias = TRUE_BIAS.get(scenario_name, 1.0)

        bias_matrix    = np.array([tr["bias_factor"].values for tr in traces])
        pstd_matrix    = np.array([tr["posterior_std"].values for tr in traces])
        pct_matrix     = np.array([tr["percentile"].values for tr in traces])
        met_matrix     = np.array([tr["deadline_met"].values for tr in traces])

        mean_pstd = np.nanmean(pstd_matrix, axis=0)
        conv_mask = mean_pstd < 0.05
        conv_step = int(np.argmax(conv_mask)) if conv_mask.any() else n_tasks

        bias_mean  = np.nanmean(bias_matrix, axis=0)
        rmse_early = float(np.sqrt(np.nanmean((bias_mean[:cutoff_early] - true_bias) ** 2)))
        rmse_late  = float(np.sqrt(np.nanmean((bias_mean[cutoff_late:]  - true_bias) ** 2)))

        deadline_at_50 = float(np.nanmean(met_matrix[:, step_50pct])) \
            if step_50pct < met_matrix.shape[1] else np.nan

        pct_range = float(np.nanmax(np.nanmean(pct_matrix, axis=0))
                          - np.nanmin(np.nanmean(pct_matrix, axis=0)))

        rows.append({
            "scenario":              scenario_name,
            "true_bias":             true_bias,
            "convergence_step":      conv_step,
            "convergence_pct":       round(conv_step / n_tasks, 2),
            "bias_rmse_early":       round(rmse_early, 4),
            "bias_rmse_late":        round(rmse_late, 4),
            "deadline_met_at_50pct": round(deadline_at_50, 3),
            "percentile_range":      round(pct_range, 4),
        })

    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# Deadline reachability phase diagram
# ---------------------------------------------------------------------------

def run_and_plot_reachability(
    task_file: str = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
    min_percentile: float = 0.05,
    n_time: int = 25,
    n_bias: int = 25,
    bias_range: tuple[float, float] = (0.5, 2.5),
    fixed_deadline: float | None = None,
    tolerance: float = 0.05,
    seed: int = PLAN_SEED,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> None:
    """
    Compute and save two deadline reachability phase diagrams.

    **Diagram 1 — replanning diagram** (``reachability.png``):
      Shows whether the deadline can be met depending on the replanning
      strategy, using the planned project duration as the deadline.

      - Green:  reachable without changing the planning percentile.
      - Yellow: reachable only by lowering the planning percentile.
      - Red:    unreachable regardless of replanning.

    **Diagram 2 — fixed-deadline diagram** (``reachability_fixed.png``):
      Shows proximity to a fixed external deadline with a tolerance band.
      Uses the most optimistic percentile (``min_percentile``) to find the
      best-case finish time, then classifies each cell.

      - Green:  best-case finish ≤ ``fixed_deadline``.
      - Yellow: within ``±tolerance`` of ``fixed_deadline``.
      - Red:    clearly missed (> ``fixed_deadline × (1 + tolerance)``).

    Args:
        task_file:        Path to tasks CSV.
        base_percentile:  Baseline planning percentile (also sets the deadline
                          for diagram 1 and, if ``fixed_deadline`` is None,
                          for diagram 2 as well).
        min_percentile:   Most optimistic replanning percentile.
        n_time:           Grid resolution along the time-fraction axis.
        n_bias:           Grid resolution along the bias axis.
        bias_range:       (min_bias, max_bias) for the y-axis.
        fixed_deadline:   Explicit deadline in days for diagram 2.
                          ``None`` → uses planned_duration (same as diagram 1).
        tolerance:        Relative margin above ``fixed_deadline`` that is still
                          considered acceptable (default 0.05 = 5 %).
        seed:             RNG seed for the baseline plan.
        output_dir:       Directory for output images.
        verbose:          Print progress.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    time_fractions = np.linspace(0.0, 1.0, n_time)
    bias_values    = np.linspace(bias_range[0], bias_range[1], n_bias)
    total_cells    = n_time * n_bias

    # ------------------------------------------------------------------ #
    # Diagram 1: replanning phase diagram                                 #
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"Computing replanning reachability grid "
              f"({n_time}×{n_bias} = {total_cells} cells)…")

    grid_opt, grid_base, planned_duration, deadline = compute_reachability_grid(
        task_file=task_file,
        time_fractions=time_fractions,
        bias_values=bias_values,
        base_percentile=base_percentile,
        min_percentile=min_percentile,
        seed=seed,
    )

    plot_deadline_reachability(
        grid_opt=grid_opt,
        grid_base=grid_base,
        time_fractions=time_fractions,
        bias_values=bias_values,
        planned_duration=planned_duration,
        deadline=deadline,
        save_path=str(output_dir / "reachability.png"),
        base_percentile=base_percentile,
        min_percentile=min_percentile,
    )

    if verbose:
        print(f"  Reachable at min p:   {grid_opt.sum()}/{total_cells} cells "
              f"({grid_opt.sum() / total_cells:.0%})")
        print(f"  Reachable at base p:  {grid_base.sum()}/{total_cells} cells "
              f"({grid_base.sum() / total_cells:.0%})")

    # ------------------------------------------------------------------ #
    # Diagram 2: fixed-deadline diagram with tolerance band               #
    # ------------------------------------------------------------------ #
    if verbose:
        dl_label = f"{fixed_deadline:.1f} d" if fixed_deadline else "planned"
        print(f"Computing fixed-deadline reachability grid "
              f"(deadline={dl_label}, tolerance=±{tolerance:.0%})…")

    zones, _, deadline_used = compute_reachability_zones(
        task_file=task_file,
        time_fractions=time_fractions,
        bias_values=bias_values,
        deadline=fixed_deadline,
        base_percentile=base_percentile,
        min_percentile=min_percentile,
        tolerance=tolerance,
        seed=seed,
    )

    plot_reachability_with_tolerance(
        zones=zones,
        time_fractions=time_fractions,
        bias_values=bias_values,
        planned_duration=planned_duration,
        deadline=deadline_used,
        save_path=str(output_dir / "reachability_fixed.png"),
        base_percentile=base_percentile,
        min_percentile=min_percentile,
        tolerance=tolerance,
    )

    if verbose:
        n_green  = (zones == 2).sum()
        n_yellow = (zones == 1).sum()
        n_red    = (zones == 0).sum()
        print(f"  Green  (achieved):   {n_green}/{total_cells} ({n_green/total_cells:.0%})")
        print(f"  Yellow (±{tolerance:.0%} margin): "
              f"{n_yellow}/{total_cells} ({n_yellow/total_cells:.0%})")
        print(f"  Red    (missed):     {n_red}/{total_cells} ({n_red/total_cells:.0%})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Convergence analysis — replanning after every task")
    print("=" * 60)

    results, target_finish, n_tasks = run_all_convergence(verbose=True)

    print("\nPlotting individual scenario convergence curves...")
    plot_convergence_grid(results, target_finish, n_tasks)

    print("\nPlotting cross-scenario summary...")
    plot_convergence_summary(results, target_finish, n_tasks)

    print("\nComputing convergence statistics...")
    stats = compute_convergence_stats(results, n_tasks, target_finish)

    stats_path = OUTPUT_DIR / "convergence_stats.csv"
    stats.to_csv(stats_path)
    print(f"\nConvergence stats → {stats_path}")

    print("\n=== Convergence statistics ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(stats.to_string())

    print("\n--- Interpretation ---")
    print("convergence_step : first task at which posterior_std < 0.05")
    print("bias_rmse_early  : how wrong the bias estimate is in the first 20% of tasks")
    print("bias_rmse_late   : how wrong it is in the last 30% (should be much smaller)")
    print("percentile_range : how much the chosen percentile shifts (larger = more adaptation)")
    print("\nReplanning after every task IS useful when:")
    print("  bias_rmse_late << bias_rmse_early  (estimate improves)")
    print("  convergence_pct < 0.5              (converges before halfway)")
    print("  percentile_range > 0.05            (replanner actually adapts)")

    print("\nBuilding deadline reachability phase diagram...")
    run_and_plot_reachability(verbose=True)
