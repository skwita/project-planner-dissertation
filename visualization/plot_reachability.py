"""Phase diagram: deadline reachability as a function of project state and bias."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_deadline_reachability(
    grid_opt: np.ndarray,
    grid_base: np.ndarray,
    time_fractions: np.ndarray,
    bias_values: np.ndarray,
    planned_duration: float,
    deadline: float,
    save_path: str,
    base_percentile: float = 0.80,
    min_percentile: float = 0.05,
) -> None:
    """
    Three-zone phase diagram of deadline reachability.

    Axes:
      - X: fraction of planned project duration elapsed (0 = start, 1 = planned end).
      - Y: bias factor — ratio of actual to planned task duration.
            Values > 1 indicate the project is running slow; < 1 means running fast.

    Zones:
      - Green  — deadline reachable without changing the planning percentile
                 (``grid_base = True``).
      - Yellow — deadline reachable only by lowering the planning percentile
                 (``grid_opt = True``, ``grid_base = False``).
      - Red    — deadline impossible regardless of replanning
                 (``grid_opt = False``).

    Phase boundaries are drawn as bold contour lines.

    Args:
        grid_opt:         Boolean matrix (n_time × n_bias); True = reachable at
                          ``min_percentile``.
        grid_base:        Boolean matrix (n_time × n_bias); True = reachable at
                          ``base_percentile``.
        time_fractions:   1-D array of project-time fractions (x-axis values).
        bias_values:      1-D array of bias factors (y-axis values).
        planned_duration: Planned project duration in days (shown in title).
        deadline:         Hard deadline in days (shown in title).
        save_path:        Output image path.
        base_percentile:  Baseline planning percentile (used in labels).
        min_percentile:   Optimistic planning percentile (used in labels).
    """
    # --- Build a combined integer zone matrix ---
    # 0 = red (unreachable), 1 = yellow (needs replanning), 2 = green (OK as-is)
    zone = np.where(grid_base, 2, np.where(grid_opt, 1, 0)).astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Pixel-aligned colour mesh
    cmap = plt.matplotlib.colors.ListedColormap(["#d73027", "#fee090", "#1a9850"])
    mesh = ax.pcolormesh(
        time_fractions, bias_values, zone.T,
        cmap=cmap, vmin=0, vmax=2, shading="nearest",
    )

    # --- Phase boundary: green / yellow ---
    if grid_base.any() and not grid_base.all():
        ax.contour(
            time_fractions, bias_values, grid_base.T.astype(float),
            levels=[0.5], colors="darkgreen", linewidths=2.0, linestyles="-",
        )

    # --- Phase boundary: yellow / red ---
    if grid_opt.any() and not grid_opt.all():
        ax.contour(
            time_fractions, bias_values, grid_opt.T.astype(float),
            levels=[0.5], colors="darkred", linewidths=2.0, linestyles="--",
        )

    # --- Reference lines ---
    ax.axhline(1.0, color="white", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.text(
        time_fractions[-1] * 0.98, 1.0 + (bias_values[-1] - bias_values[0]) * 0.02,
        "no bias (1×)", color="white", fontsize=8, ha="right", va="bottom",
    )

    ax.set_xlabel("Project time elapsed  (fraction of planned duration)", fontsize=12)
    ax.set_ylabel("Bias factor  (actual / planned task duration)", fontsize=12)
    ax.set_title(
        f"Deadline reachability phase diagram\n"
        f"planned = {planned_duration:.1f} d  |  deadline = {deadline:.1f} d  |  "
        f"base p = {base_percentile:.0%}  |  min p = {min_percentile:.0%}",
        fontsize=11,
    )

    legend_patches = [
        mpatches.Patch(color="#1a9850",
                       label=f"Reachable — no replanning needed  (p = {base_percentile:.0%})"),
        mpatches.Patch(color="#fee090",
                       label=f"Reachable — lower percentile required  (p ≥ {min_percentile:.0%})"),
        mpatches.Patch(color="#d73027",
                       label=f"Unreachable — deadline impossible  (even at p = {min_percentile:.0%})"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
              framealpha=0.85)

    ax.set_xlim(time_fractions[0], time_fractions[-1])
    ax.set_ylim(bias_values[0], bias_values[-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_reachability_with_tolerance(
    zones: np.ndarray,
    time_fractions: np.ndarray,
    bias_values: np.ndarray,
    planned_duration: float,
    deadline: float,
    save_path: str,
    base_percentile: float = 0.80,
    min_percentile: float = 0.05,
    tolerance: float = 0.05,
) -> None:
    """
    Fixed-deadline reachability diagram with a tolerance band.

    Axes:
      - X: fraction of planned project duration elapsed.
      - Y: bias factor (actual / planned task duration).

    Zones (classified by the earliest achievable finish at ``min_percentile``):
      - Green  — finish ≤ deadline (deadline achieved).
      - Yellow — deadline < finish ≤ deadline × (1 + tolerance)
                 (missed, but within the ±``tolerance`` margin).
      - Red    — finish > deadline × (1 + tolerance)  (clearly missed).

    Two phase-boundary contours are drawn:
      - Solid dark-green line:   green / yellow boundary (deadline itself).
      - Dashed dark-red line:    yellow / red boundary (deadline + tolerance).

    Args:
        zones:            Integer matrix (n_time × n_bias) with values 0/1/2
                          from ``compute_reachability_zones``.
        time_fractions:   1-D array of project-time fractions (x-axis).
        bias_values:      1-D array of bias factors (y-axis).
        planned_duration: Planned project duration in days.
        deadline:         Fixed external deadline in days.
        save_path:        Output image path.
        base_percentile:  Percentile used for the baseline plan (title only).
        min_percentile:   Optimistic planning percentile (title only).
        tolerance:        Relative tolerance above deadline (default 0.05).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.matplotlib.colors.ListedColormap(["#d73027", "#fee090", "#1a9850"])
    ax.pcolormesh(
        time_fractions, bias_values, zones.T.astype(float),
        cmap=cmap, vmin=0, vmax=2, shading="nearest",
    )

    z = zones.T.astype(float)

    # --- Green / yellow boundary: finish == deadline ---
    if (zones == 2).any() and (zones < 2).any():
        ax.contour(
            time_fractions, bias_values, z,
            levels=[1.5], colors="darkgreen", linewidths=2.5, linestyles="-",
        )

    # --- Yellow / red boundary: finish == deadline * (1 + tolerance) ---
    if (zones >= 1).any() and (zones == 0).any():
        ax.contour(
            time_fractions, bias_values, z,
            levels=[0.5], colors="darkred", linewidths=2.5, linestyles="--",
        )

    # --- Reference: no-bias line ---
    ax.axhline(1.0, color="white", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.text(
        time_fractions[-1] * 0.98,
        1.0 + (bias_values[-1] - bias_values[0]) * 0.02,
        "no bias (1×)", color="white", fontsize=8, ha="right", va="bottom",
    )

    ax.set_xlabel("Project time elapsed  (fraction of planned duration)", fontsize=12)
    ax.set_ylabel("Bias factor  (actual / planned task duration)", fontsize=12)

    overage = f"+{tolerance:.0%}"
    ax.set_title(
        f"Deadline reachability — fixed deadline with {overage} tolerance band\n"
        f"planned = {planned_duration:.1f} d  |  deadline = {deadline:.1f} d  |  "
        f"base p = {base_percentile:.0%}  |  min p = {min_percentile:.0%}",
        fontsize=11,
    )

    legend_patches = [
        mpatches.Patch(color="#1a9850",
                       label=f"Achieved — finish ≤ {deadline:.1f} d"),
        mpatches.Patch(color="#fee090",
                       label=f"Marginal — within {overage} of deadline "
                             f"(≤ {deadline * (1 + tolerance):.1f} d)"),
        mpatches.Patch(color="#d73027",
                       label=f"Missed — finish > {deadline * (1 + tolerance):.1f} d"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.85)

    ax.set_xlim(time_fractions[0], time_fractions[-1])
    ax.set_ylim(bias_values[0], bias_values[-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
