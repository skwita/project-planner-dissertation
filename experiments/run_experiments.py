"""
Replanning experiment runner.

Runs every scenario from ``scenario_generator.SCENARIOS`` across multiple
observation snapshots (how many tasks are already done) and multiple random
seeds, collects replanning results, and exports:

  - ``output/experiments/results.csv``   — full row-per-run table
  - ``output/experiments/summary.xlsx``  — pivot summary per scenario
  - ``output/experiments/plots/``        — one comparison plot per scenario

Usage
-----
    python -m experiments.run_experiments

or import and call ``run_all_experiments()`` directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure the project root is on the path when running as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.scenario_generator import SCENARIOS, load_scheduled_tasks
from services.bayes_rescheduler import reschedule_with_fixed_project_deadline
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_FILE = "data/tasks.csv"
BASE_PERCENTILE = 0.9
PLAN_SEED = 42

# What fraction of tasks are already observed at each snapshot
OBSERVATION_FRACTIONS = [0.25, 0.5, 0.75]

# Number of random seeds per (scenario, snapshot) combination
N_SEEDS = 5

OUTPUT_DIR = Path("output/experiments")


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_single(
    scenario_name: str,
    n_observed: int,
    seed: int,
    scheduled_tasks,
    target_finish_time: float,
    task_file: str = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
) -> dict:
    """
    Run one (scenario, snapshot, seed) combination.

    ``scheduled_tasks`` is the baseline plan (planned_* fields set), used by
    scenario generators to sort tasks by planned_end_time.  A fresh
    unscheduled copy is loaded for the rescheduler so it starts clean.
    """
    scenario = SCENARIOS[scenario_name]
    desc = scenario["description"]
    n_tasks = len(scheduled_tasks)

    # Generate progress dict using the scheduled plan for ordering
    progress = scenario["fn"](
        scheduled_tasks, n_observed=n_observed, seed=seed, **scenario["kwargs"]
    )

    # Fresh tasks with the same baseline plan for the rescheduler.
    # Must be scheduled so that planned_start_time is available for the
    # factual schedule reconstruction (plan-driven start-time floors).
    from services.parser import load_tasks_from_csv as _load
    fresh_tasks = _load(task_file)
    build_schedule(fresh_tasks, percentile=base_percentile, seed=PLAN_SEED)

    base_row = {
        "scenario": scenario_name,
        "description": desc,
        "n_observed": n_observed,
        "obs_fraction": round(n_observed / n_tasks, 2),
        "seed": seed,
    }

    try:
        result = reschedule_with_fixed_project_deadline(
            tasks=fresh_tasks,
            progress=progress,
            target_finish_time=target_finish_time,
            base_percentile=base_percentile,
        )
    except Exception as exc:
        return {
            **base_row,
            "error": str(exc),
            "used_percentile": np.nan,
            "bias_factor": np.nan,
            "project_finish": np.nan,
            "deadline_met": False,
            "used_observations": 0,
            "finish_vs_deadline": np.nan,
        }

    return {
        **base_row,
        "error": None,
        "used_percentile": round(result["used_percentile"], 4),
        "bias_factor": round(result["bias_factor_mean"], 4),
        "project_finish": round(result["project_finish"], 2),
        "deadline_met": result["deadline_met"],
        "used_observations": result["used_observations"],
        "finish_vs_deadline": round(
            result["project_finish"] - target_finish_time, 2
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all_experiments(
    task_file: str = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
    plan_seed: int = PLAN_SEED,
    observation_fractions: list[float] = OBSERVATION_FRACTIONS,
    n_seeds: int = N_SEEDS,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all scenarios and collect results.

    Args:
        task_file:             Path to the tasks CSV.
        base_percentile:       Percentile for the baseline plan.
        plan_seed:             Seed for the baseline plan (fixed).
        observation_fractions: Fractions of tasks marked as "done".
        n_seeds:               Number of random seeds per run.
        output_dir:            Directory for output files.
        verbose:               Print progress to stdout.

    Returns:
        DataFrame with one row per (scenario, obs_fraction, seed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Build the one baseline plan that defines the deadline
    tasks = load_scheduled_tasks(task_file, base_percentile, plan_seed)
    n_tasks = len(tasks)
    target_finish = max(t.planned_end_time for t in tasks)

    if verbose:
        print(f"Baseline plan: {n_tasks} tasks, deadline = {target_finish:.1f} days")
        print(f"Scenarios: {len(SCENARIOS)}, "
              f"snapshots: {len(observation_fractions)}, "
              f"seeds: {n_seeds}")
        print(f"Total runs: "
              f"{len(SCENARIOS) * len(observation_fractions) * n_seeds}\n")

    # Compute n_observed values from fractions
    obs_counts = [max(1, round(f * n_tasks)) for f in observation_fractions]

    rows = []
    total = len(SCENARIOS) * len(obs_counts) * n_seeds
    done = 0

    first_error_shown = False
    for scenario_name in SCENARIOS:
        for n_obs in obs_counts:
            for seed in range(n_seeds):
                row = run_single(
                    scenario_name=scenario_name,
                    n_observed=n_obs,
                    seed=seed,
                    scheduled_tasks=tasks,
                    target_finish_time=target_finish,
                    task_file=task_file,
                    base_percentile=base_percentile,
                )
                rows.append(row)
                done += 1
                # Show first error in full to help debugging
                if row.get("error") and not first_error_shown:
                    print(f"  [WARN] First error in {scenario_name} "
                          f"obs={n_obs} seed={seed}: {row['error']}")
                    first_error_shown = True
                if verbose and done % 10 == 0:
                    print(f"  {done}/{total} — {scenario_name} "
                          f"obs={n_obs} seed={seed}")

    df = pd.DataFrame(rows)

    # Diagnostics: show error rate
    n_errors = df["error"].notna().sum()
    if n_errors:
        print(f"\n[WARN] {n_errors}/{len(df)} runs had errors. "
              f"Check output/experiments/results.csv for details.")
    else:
        print(f"\nAll {len(df)} runs completed successfully.")

    # ---- Export raw results ----
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"Raw results → {csv_path}")

    # ---- Summary table ----
    _export_summary(df, output_dir, verbose)

    # ---- Plots ----
    _plot_all_scenarios(df, output_dir, target_finish, verbose)

    return df


# ---------------------------------------------------------------------------
# Summary export
# ---------------------------------------------------------------------------

def _export_summary(df: pd.DataFrame, output_dir: Path, verbose: bool) -> None:
    """Build and save a pivot summary per scenario × obs_fraction."""
    numeric_cols = ["used_percentile", "bias_factor", "project_finish",
                    "finish_vs_deadline", "used_observations"]

    summary = (
        df[df["error"].isna()]
        .groupby(["scenario", "obs_fraction"])[numeric_cols]
        .agg(["mean", "std"])
        .round(3)
    )
    # Flatten multi-level columns
    summary.columns = ["_".join(c) for c in summary.columns]

    deadline_rate = (
        df[df["error"].isna()]
        .groupby(["scenario", "obs_fraction"])["deadline_met"]
        .mean()
        .rename("deadline_met_rate")
    )
    summary = summary.join(deadline_rate)

    xlsx_path = output_dir / "summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary")
        df.to_excel(writer, sheet_name="Raw", index=False)

        # One sheet per scenario for easy inspection
        for name in df["scenario"].unique():
            sub = df[df["scenario"] == name]
            sheet = name[:31]  # Excel sheet name limit
            sub.to_excel(writer, sheet_name=sheet, index=False)

    if verbose:
        print(f"Summary      → {xlsx_path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_all_scenarios(
    df: pd.DataFrame,
    output_dir: Path,
    target_finish: float,
    verbose: bool,
) -> None:
    """One 2×2 summary figure per scenario."""
    plots_dir = output_dir / "plots"
    clean = df[df["error"].isna()].copy()

    for scenario_name in SCENARIOS:
        sub = clean[clean["scenario"] == scenario_name]
        if sub.empty:
            continue

        desc = SCENARIOS[scenario_name]["description"]
        fractions = sorted(sub["obs_fraction"].unique())

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{scenario_name}\n{desc}", fontsize=11, fontweight="bold")

        # Panel 1: chosen percentile vs obs_fraction
        ax = axes[0, 0]
        _boxplot_by_fraction(ax, sub, "used_percentile", fractions,
                             "Chosen percentile", BASE_PERCENTILE,
                             hline_label="base percentile")

        # Panel 2: bias factor vs obs_fraction
        ax = axes[0, 1]
        _boxplot_by_fraction(ax, sub, "bias_factor", fractions,
                             "Bias factor (exp θ)", 1.0,
                             hline_label="no bias")

        # Panel 3: project finish vs obs_fraction
        ax = axes[1, 0]
        _boxplot_by_fraction(ax, sub, "project_finish", fractions,
                             "Replanned finish (days)", target_finish,
                             hline_label="deadline", hline_color="red")

        # Panel 4: deadline met rate (bar chart)
        ax = axes[1, 1]
        rates = [
            sub[sub["obs_fraction"] == f]["deadline_met"].mean()
            for f in fractions
        ]
        bars = ax.bar([str(f) for f in fractions], rates,
                      color=["green" if r >= 0.8 else "orange" if r >= 0.5
                             else "red" for r in rates])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Observation fraction")
        ax.set_ylabel("Deadline met rate")
        ax.set_title("Deadline met rate")
        ax.axhline(0.8, linestyle="--", color="green", alpha=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{rate:.0%}", ha="center", fontsize=9)

        plt.tight_layout()
        safe_name = scenario_name.replace("/", "_").replace(".", "_")
        fig_path = plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    if verbose:
        print(f"Plots        → {plots_dir}/")


def _boxplot_by_fraction(
    ax, sub: pd.DataFrame, col: str, fractions: list,
    ylabel: str, hline_val: float,
    hline_label: str = "", hline_color: str = "gray",
) -> None:
    """Draw a boxplot of ``col`` grouped by obs_fraction on ``ax``."""
    data = [sub[sub["obs_fraction"] == f][col].dropna().values
            for f in fractions]
    ax.boxplot(data, labels=[str(f) for f in fractions], patch_artist=True)
    ax.axhline(hline_val, linestyle="--", color=hline_color,
               alpha=0.7, label=hline_label)
    ax.set_xlabel("Observation fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if hline_label:
        ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Scenario comparison plot (across all scenarios at fixed obs_fraction)
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    df: pd.DataFrame,
    obs_fraction: float = 0.5,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Heatmap: scenarios (rows) × metrics (columns), averaged over seeds.

    Each column is normalised independently to [0, 1] before colouring so
    that the RdYlGn colourmap is meaningful across metrics with different
    scales.  Raw values are shown as text inside each cell.
    """
    output_dir = Path(output_dir)

    clean = df[(df["error"].isna()) & (df["obs_fraction"] == obs_fraction)]
    if clean.empty:
        print(f"[WARN] plot_scenario_comparison: no clean rows for "
              f"obs_fraction={obs_fraction}. Skipping.")
        return

    overview = pd.concat([
        clean.groupby("scenario")["used_percentile"].mean().rename("Percentile"),
        clean.groupby("scenario")["bias_factor"].mean().rename("Bias factor"),
        clean.groupby("scenario")["finish_vs_deadline"].mean().rename("Finish-deadline"),
        clean.groupby("scenario")["deadline_met"].mean().rename("Deadline met %"),
    ], axis=1).round(3)

    if overview.empty:
        print(f"[WARN] plot_scenario_comparison: overview is empty. Skipping.")
        return

    # Normalise each column to [0, 1] for colouring only
    normed = overview.copy()
    for col in normed.columns:
        col_min, col_max = normed[col].min(), normed[col].max()
        if col_max > col_min:
            normed[col] = (normed[col] - col_min) / (col_max - col_min)
        else:
            normed[col] = 0.5  # all identical — mid-colour

    fig, ax = plt.subplots(figsize=(10, max(4, len(overview) * 0.55)))
    im = ax.imshow(normed.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(overview.columns)))
    ax.set_xticklabels(overview.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(overview)))
    ax.set_yticklabels(overview.index, fontsize=8)

    # Annotate with raw values
    for i in range(len(overview)):
        for j, col in enumerate(overview.columns):
            val = overview.iloc[i, j]
            fmt = f"{val:.0%}" if col == "Deadline met %" else f"{val:.3f}"
            brightness = normed.iloc[i, j]
            text_color = "black" if 0.2 < brightness < 0.8 else "white"
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")

    ax.set_title(
        f"Scenario comparison — obs_fraction = {obs_fraction:.0%}  "
        f"(colour = normalised per column)",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = output_dir / f"scenario_comparison_{obs_fraction:.0%}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison heatmap → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_all_experiments()

    print("\n=== Top results by scenario (obs_fraction=0.5) ===")
    mid = df[(df["obs_fraction"] == 0.5) & df["error"].isna()]
    summary = (mid.groupby("scenario")
               .agg(
                   bias_mean=("bias_factor", "mean"),
                   percentile_mean=("used_percentile", "mean"),
                   finish_mean=("project_finish", "mean"),
                   deadline_rate=("deadline_met", "mean"),
               )
               .round(3)
               .sort_values("deadline_rate", ascending=False))
    print(summary.to_string())

    plot_scenario_comparison(df, obs_fraction=0.5)
    plot_scenario_comparison(df, obs_fraction=0.25)
    plot_scenario_comparison(df, obs_fraction=0.75)
