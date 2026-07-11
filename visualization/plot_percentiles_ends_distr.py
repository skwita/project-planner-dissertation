"""PDF and CDF plots for project duration distributions across percentiles."""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_COLORS = ["skyblue", "orange", "green", "red", "purple"]


def plot_percentile_pdf(
    durations_list: list[np.ndarray],
    labels: list,
    filename: str,
    bins: int = 40,
    xlim: tuple[float, float] = (40, 80),
) -> None:
    """
    Plot histogram + KDE density for one or more duration distributions.

    The KDE is fitted to the full data but rendered only from the right
    edge of the second histogram bin onward (to suppress near-zero artefacts
    from degenerate short-duration samples).

    Args:
        durations_list: One array per percentile/scenario.
        labels:         Legend labels (same length as durations_list).
        filename:       Output image path.
        bins:           Number of histogram bins.
        xlim:           X-axis limits used for both histogram and KDE grid.
    """
    plt.figure(figsize=(12, 8))
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    # kde_start = bin_edges[2]  # skip first two bins
    # x_grid = np.linspace(kde_start, xlim[1], 1000)

    for i, durations in enumerate(durations_list):
        durations = np.asarray(durations)
        color = _COLORS[i % len(_COLORS)]

        plt.hist(durations, bins=bins, range=xlim, density=True,
                 alpha=0.4, color=color, edgecolor="black",
                 label=f"{labels[i]} (hist.)")

        # valid = durations[durations >= kde_start]
        # if valid.size > 1:
            # kde = gaussian_kde(durations)
            # y = kde(x_grid)
            # Scale KDE so its area over [kde_start, xlim[1]] matches the
            # proportion of data in that region
            # mass = np.mean(durations >= kde_start)
            # area = np.trapezoid(y, x_grid)
            # if area > 0:
                # y = y * mass / area
            # line, = plt.plot(x_grid, y, color=color, linewidth=2.5,
                            #  label=str(labels[i]))
            # line.set_path_effects(
                # [pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
            # )
        # else:
            # print(f"  ⚠ Too few points for KDE: {labels[i]}")

    plt.xlabel("Project duration (days)")
    plt.ylabel("Probability density")
    plt.title("Project duration PDF")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_percentile_cdfs(
    durations_list: list[np.ndarray],
    labels: list,
    filename: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Plot empirical CDFs for one or more duration distributions.

    Args:
        durations_list: One array per percentile/scenario.
        labels:         Legend labels (same length as durations_list).
        filename:       Output image path.
        xlim:           Optional X-axis limits.
    """
    plt.figure(figsize=(12, 8))

    for i, durations in enumerate(durations_list):
        sorted_d = np.sort(durations)
        cdf_vals = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        color = _COLORS[i % len(_COLORS)]
        # Convert label to string; callers sometimes pass raw floats
        plt.plot(sorted_d, cdf_vals, color=color, linewidth=2.5,
                 label=str(labels[i]))

    if xlim:
        plt.xlim(xlim)

    plt.xlabel("Project duration (days)")
    plt.ylabel("Cumulative probability (CDF)")
    plt.title("Project duration CDF")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
