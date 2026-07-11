"""Pareto scatter plot: average project duration vs. average idle time."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _hyperbolic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Hyperbolic curve: a / (x - b) + c."""
    return a / (x - b) + c


def plot_idle_vs_duration(
    durations: list[float],
    idles_sum: list[float],
    percentiles_tasks: list[float],
    n_iter: int,
    save_path: str,
    max_duration: float | None = None,
) -> None:
    """
    Pareto scatter: mean project duration (x) vs. mean total idle time (y).

    Each point is labelled with its task percentile (as an integer, e.g. 50).
    Points are colour-coded by percentile.  When ``max_duration`` is given,
    points to the right of that threshold are shown as grey/transparent and
    a vertical deadline line is drawn.

    A hyperbolic curve fit is attempted; the result is currently not plotted
    (uncomment the ``plt.plot`` line below to enable it).

    Args:
        durations:        Mean project duration per percentile.
        idles_sum:        Mean total idle time per percentile.
        percentiles_tasks: Percentile values matching the two lists above.
        n_iter:           Number of MC iterations (shown in the title).
        save_path:        Output image path.
        max_duration:     Optional deadline for colour-splitting the scatter.
    """
    plt.figure(figsize=(8, 6))

    np_dur = np.array(durations)
    np_idle = np.array(idles_sum)
    np_pct = np.array(percentiles_tasks)

    if max_duration is not None:
        mask = np_dur <= max_duration
        sc = plt.scatter(np_dur[mask], np_idle[mask],
                         c=np_pct[mask], cmap="viridis",
                         s=80, edgecolors="black")
        plt.scatter(np_dur[~mask], np_idle[~mask],
                    color="gray", alpha=0.4, s=80, edgecolors="black")
        cbar = plt.colorbar(sc)
        plt.axvline(max_duration, linestyle="--", color="red", linewidth=1.5)
    else:
        sc = plt.scatter(np_dur, np_idle,
                         c=np_pct, cmap="viridis",
                         s=80, edgecolors="black")
        cbar = plt.colorbar(sc)

    cbar.set_label("Task percentile")

    # Attempt hyperbolic fit (for diagnostics; not plotted by default)
    try:
        popt, _ = curve_fit(_hyperbolic, durations, idles_sum,
                            p0=(1.0, 0.01, 1.0), maxfev=10_000)
        print(f"Fit params: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
        # x_fit = np.linspace(np_dur.min(), np_dur.max(), 300)
        # plt.plot(x_fit, _hyperbolic(x_fit, *popt), "r--", linewidth=2)
    except RuntimeError:
        print("⚠  Hyperbolic fit did not converge.")

    # Point labels (percentile as integer %)
    for i, p in enumerate(np_pct):
        plt.text(np_dur[i] + 0.3, np_idle[i] + 0.3, f"{p * 100:.0f}", fontsize=8)

    plt.xlabel("Mean project duration (days)")
    plt.ylabel("Mean total idle time (days)")
    plt.title(f"Pareto: idle time vs. duration  ({n_iter:,} iterations)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pareto_transition(
    durations_base: list[float],
    idles_base: list[float],
    durations_new: list[float],
    idles_new: list[float],
    percentiles: list[float],
    deadline: float,
    save_path: str,
) -> None:
    """
    Show how the optimal operating point shifts on the Pareto front after
    Bayesian bias correction.

    Two scatter clouds are drawn (original vs. updated predictive
    distributions), the feasible region is delimited by a deadline line,
    and an arrow connects the old optimal point to the new one.

    Args:
        durations_base:  Mean project durations before replanning.
        idles_base:      Mean idle times before replanning.
        durations_new:   Mean project durations after replanning.
        idles_new:       Mean idle times after replanning.
        percentiles:     Task percentiles corresponding to each point.
        deadline:        Hard deadline (vertical line).
        save_path:       Output image path.
    """
    plt.figure(figsize=(9, 7))

    np_pct = np.array(percentiles)
    d0, i0 = np.array(durations_base), np.array(idles_base)
    d1, i1 = np.array(durations_new), np.array(idles_new)

    plt.scatter(d0, i0, c=np_pct, cmap="Blues",
                s=70, edgecolors="black", label="Original")
    plt.scatter(d1, i1, c=np_pct, cmap="Greens",
                s=70, edgecolors="black", label="After bias update")
    plt.axvline(deadline, linestyle="--", color="red",
                linewidth=2, label="Deadline")

    # Optimal point: rightmost feasible point on each curve
    def _optimal(durations: np.ndarray, idles: np.ndarray) -> tuple[float, float]:
        mask = durations <= deadline
        idx = np.where(mask)[0][np.argmax(durations[mask])] if mask.any() else int(np.argmin(durations))
        return float(durations[idx]), float(idles[idx])

    x0, y0 = _optimal(d0, i0)
    x1, y1 = _optimal(d1, i1)

    plt.scatter(x0, y0, color="blue", s=140, zorder=5)
    plt.scatter(x1, y1, color="green", s=140, zorder=5)
    plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", color="black", lw=2))
    plt.text(x0 + 0.2, y0 + 0.2, "old plan", fontsize=10, color="blue")
    plt.text(x1 + 0.2, y1 + 0.2, "new plan", fontsize=10, color="green")

    for i, p in enumerate(np_pct):
        plt.text(d1[i] + 0.3, i1[i] + 0.3, f"{p:.2f}", fontsize=7)

    plt.xlabel("Mean project duration (days)")
    plt.ylabel("Mean total idle time (days)")
    plt.title("Pareto front shift after bias correction")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_history_metrics(
    history: list[dict],
    save_prefix: str = "output/plots/history",
) -> None:
    """
    Plot the evolution of key replanning metrics across iterative stages.

    Produces three separate figures:
      - Bias factor (exp θ) per stage.
      - Chosen planning percentile per stage.
      - Expected project finish time per stage.

    Args:
        history:     List of stage dicts with keys ``stage``, ``bias``,
                     ``percentile``, and ``finish``.
        save_prefix: Path prefix; ``_bias.png``, ``_percentile.png``,
                     and ``_finish.png`` are appended automatically.
    """
    stages = [h["stage"] for h in history]
    metrics = [
        ("bias",       "Bias factor (exp θ)",   "Bias dynamics"),
        ("percentile", "Planning percentile",    "Percentile dynamics"),
        ("finish",     "Project finish (days)",  "Finish time dynamics"),
    ]
    for key, ylabel, title in metrics:
        values = [h[key] for h in history]
        plt.figure()
        plt.plot(stages, values, marker="o")
        plt.xlabel("Stage")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{key}.png", dpi=300)
        plt.close()
    print(f"History plots saved with prefix: {save_prefix}")
