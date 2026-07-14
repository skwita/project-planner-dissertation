"""Pareto scatter plot: average project duration vs. average idle time."""

import matplotlib
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


def plot_pareto_shift_trajectory(
    baseline_durations: list[float],
    baseline_idles: list[float],
    points: list[dict],
    deadline: float,
    save_path: str,
    show: bool = False,
) -> None:
    """
    Show the deadline-drift / percentile-compensation cycle during
    iterative replanning, overlaid on the original duration/idle Pareto front.

    ``points`` is the output of ``main.build_pareto_shift_updates``: a
    single ``"baseline"`` point followed by ``n_updates`` (``"drift"``,
    ``"compensate"``) pairs, one pair per data reveal. Two segment types
    are drawn, alternating:

      - **Drift** (orange): as newly observed actuals refine the duration
        bias, the expected finish time at the *unchanged* base percentile
        moves away from the deadline — the plan drifts off the original
        Pareto front, primarily along the duration axis.
      - **Compensate** (firebrick): the planning percentile is searched to
        pull the finish time back toward the deadline. Since percentile is
        exactly the parameter that traces the Pareto front, this move is
        diagonal — part of the drift is undone, at the cost (or benefit)
        of changed resource idle time.

    If a compensate point carries a ``median_feasible`` flag (see
    ``build_pareto_shift_updates``), its marker edge is colour-coded:
    green when the deadline is still achievable at the bias-corrected
    median (a genuine, sustainable correction), red when it isn't (the
    percentile is being forced below the median — a bet that a purely
    deterministic bias will never pay off, foreshadowing further drift).

    Point labels are offset in screen space (not data space) and staggered
    by update parity so that clustered points — common once the trajectory
    settles near the deadline — don't render one on top of another; each
    label also gets a translucent white backing so it stays legible over
    the arrows and other markers.

    Args:
        baseline_durations: Mean project duration per percentile (original
                            sweep, e.g. from ``compute_pareto_idle_duration_curve``).
        baseline_idles:     Mean total idle time per percentile, parallel
                            to ``baseline_durations``.
        points:             Point list from ``build_pareto_shift_updates``;
                            each dict has ``kind``, ``update``, ``finish``,
                            ``effort`` and ``percentile``. Compensate points
                            may additionally carry ``median_feasible``.
        deadline:           Hard deadline (vertical reference line).
        save_path:          Output image path.
        show:               If True, also open the figure in an interactive
                            window (switches off the ``Agg`` backend for
                            this call if needed) and block until it's closed.
    """
    # Switch to an interactive backend *before* the figure is created — once
    # a figure exists under a non-GUI backend like Agg, switching later
    # won't give it a window to draw into.
    if show and matplotlib.get_backend().lower() == "agg":
        plt.switch_backend("TkAgg")

    plt.figure(figsize=(11, 8.5))

    bd = np.asarray(baseline_durations, dtype=float)
    bi = np.asarray(baseline_idles, dtype=float)
    order = np.argsort(bd)
    plt.plot(bd[order], bi[order], "o--", color="grey", alpha=0.6,
             linewidth=1.5, markersize=5, zorder=1,
             label="Baseline plan (original Pareto front)")

    prev = points[0]
    plt.scatter([prev["finish"]], [prev["effort"]], color="steelblue", s=120,
                zorder=5, edgecolors="black", marker="s", label="Baseline point")

    label_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="none", alpha=0.82)
    leader_line = dict(arrowstyle="-", color="0.45", lw=0.6, alpha=0.7,
                        shrinkA=0, shrinkB=3)
    # Once a trajectory settles, several consecutive points can land almost
    # on top of each other (e.g. a percentile pinned at a boundary for
    # several updates). A simple up/down alternation isn't enough to keep
    # their labels apart, so cycle through a ring of offset directions —
    # one new slot per label placed, expanding the ring radius every full
    # lap — and draw a thin leader line from each label back to its point.
    _label_slots = [
        (12, 0), (12, 24), (-12, 24), (-70, 0), (-70, -24), (12, -24),
        (12, 48), (-90, 48), (-90, -48), (12, -48),
    ]
    label_counter = 0

    def _next_offset() -> tuple[float, float]:
        nonlocal label_counter
        dx, dy = _label_slots[label_counter % len(_label_slots)]
        lap = label_counter // len(_label_slots)
        label_counter += 1
        return dx * (1 + 0.6 * lap), dy * (1 + 0.6 * lap)

    drift_labelled = False
    compensate_labelled = False

    for pt in points[1:]:
        x0, y0 = prev["finish"], prev["effort"]
        x1, y1 = pt["finish"], pt["effort"]

        if pt["kind"] == "drift":
            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
            plt.scatter([x1], [y1], color="darkorange", s=90, zorder=5,
                        edgecolors="black", marker="^",
                        label=None if drift_labelled else "Drift (deadline shift, bias update)")
            drift_labelled = True
            plt.annotate(f"U{pt['update']} drift", xy=(x1, y1),
                         xytext=_next_offset(), textcoords="offset points",
                         fontsize=7, ha="left", va="center", bbox=label_bbox,
                         arrowprops=leader_line)
        else:  # compensate
            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", color="firebrick", lw=2))
            median_feasible = pt.get("median_feasible")
            edge_color = (
                "black" if median_feasible is None
                else ("seagreen" if median_feasible else "red")
            )
            edge_width = 1.0 if median_feasible is None else 2.5
            plt.scatter([x1], [y1], color="firebrick", s=110, zorder=6,
                        edgecolors=edge_color, linewidths=edge_width, marker="o",
                        label=None if compensate_labelled else "Compensate (percentile search)")
            compensate_labelled = True
            flag = "" if median_feasible is None else (" ✓med" if median_feasible else " ✗med")
            plt.annotate(f"U{pt['update']}: p={pt['percentile']:.2f}{flag}", xy=(x1, y1),
                         xytext=_next_offset(), textcoords="offset points",
                         fontsize=7, ha="left", va="center", fontweight="bold",
                         bbox=label_bbox, arrowprops=leader_line)

        prev = pt

    plt.axvline(deadline, linestyle="--", color="black", linewidth=1.5, label="Deadline")

    plt.xlabel("Project duration / срок (days)")
    plt.ylabel("Resource idle time / трудозатраты (days)")
    plt.title("Pareto front shift: deadline drift vs. percentile compensation")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    # Extra vertical breathing room so labels fanned out below a clustered
    # point don't collide with the x-axis title.
    plt.margins(y=0.18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close()


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
