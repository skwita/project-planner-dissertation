"""Gantt chart visualisations: standard and replanned."""

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from models.task import Task


def _make_color_map(tasks: list[Task]) -> dict[str, tuple]:
    """Build a role → RGBA colour mapping."""
    roles = sorted({t.role for t in tasks})
    palette = cm.get_cmap("Accent", max(len(roles), 1))
    return {role: palette(i) for i, role in enumerate(roles)}


def _sort_key(t: Task) -> tuple:
    """Sort tasks by real start, then planned start, then ID."""
    return (
        t.real_start_time if t.real_start_time is not None else 1e9,
        t.planned_start_time if t.planned_start_time is not None else 1e9,
        t.task_id,
    )


def plot_gantt(tasks: list[Task], filename: str,
               pr_buffer: float | None = None) -> None:
    """
    Save a Gantt chart showing planned (coloured) and real (black) bars.

    Bars are skipped gracefully when the corresponding time fields are None
    (e.g. for tasks that were not yet scheduled in a partial plan).
    An optional project buffer bar is appended at the bottom in red.

    Args:
        tasks:     Scheduled tasks.
        filename:  Output image path.
        pr_buffer: Project buffer size in days (omitted if None or 0).
    """
    _, ax = plt.subplots(figsize=(14, 6))
    color_map = _make_color_map(tasks)

    sorted_tasks = sorted(tasks, key=_sort_key)
    y_labels = [f"Task {t.task_id} ({t.role})" for t in sorted_tasks]
    yticks = list(range(len(sorted_tasks)))

    for i, task in enumerate(sorted_tasks):
        color = color_map[task.role]

        if task.planned_start_time is not None and task.planned_duration is not None:
            ax.barh(i, task.planned_duration, left=task.planned_start_time,
                    height=0.5, color=color, alpha=0.75)

        if task.real_start_time is not None and task.real_duration is not None:
            ax.barh(i, task.real_duration, left=task.real_start_time,
                    height=0.2, color="black")

    if pr_buffer:
        valid_ends = [t.planned_end_time for t in tasks
                      if t.planned_end_time is not None]
        if valid_ends:
            ax.barh(len(tasks), pr_buffer, left=max(valid_ends),
                    height=0.5, color="red", alpha=0.6)
            y_labels.append("Project buffer")
            yticks.append(len(tasks))

    ax.invert_yaxis()
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (days)")
    ax.set_title("Gantt Chart")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="black", label="Actual"),
        plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.6, label="Planned"),
    ])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_replanned_gantt(
    original_tasks: list[Task],
    replanned_tasks: list[Task],
    filename: str,
    target_finish_time: float | None = None,
) -> None:
    """
    Compare the original plan, the replanned schedule, and known actuals.

    Three bar layers per task:
      - Light grey (wide):   original planned duration/start.
      - Coloured (medium):   new replanned planned duration/start.
      - Black (thin):        reconstructed actual (real) duration/start.

    A red dashed vertical line marks the target deadline when provided.

    Args:
        original_tasks:     Tasks from the baseline schedule.
        replanned_tasks:    Tasks after Bayesian replanning.
        filename:           Output image path.
        target_finish_time: Deadline to draw as a vertical line.
    """
    _, ax = plt.subplots(figsize=(16, 8))

    original_map = {t.task_id: t for t in original_tasks}
    color_map = _make_color_map(replanned_tasks)

    sorted_tasks = sorted(
        replanned_tasks,
        key=lambda t: (
            t.planned_start_time if t.planned_start_time is not None else 1e9,
            t.task_id,
        ),
    )
    y_labels = [f"Task {t.task_id} ({t.role})" for t in sorted_tasks]
    yticks = list(range(len(sorted_tasks)))

    for i, task in enumerate(sorted_tasks):
        orig = original_map[task.task_id]
        color = color_map[task.role]

        # Original plan (grey background)
        if orig.planned_start_time is not None and orig.planned_duration is not None:
            ax.barh(i, orig.planned_duration, left=orig.planned_start_time,
                    height=0.55, color="lightgray", alpha=0.95)

        # Replanned schedule (coloured)
        if task.planned_start_time is not None and task.planned_duration is not None:
            ax.barh(i, task.planned_duration, left=task.planned_start_time,
                    height=0.35, color=color, alpha=0.85)

        # Actual (black, thin)
        if task.real_start_time is not None and task.real_duration is not None:
            ax.barh(i, task.real_duration, left=task.real_start_time,
                    height=0.18, color="black", alpha=1.0)

    if target_finish_time is not None:
        ax.axvline(target_finish_time, color="red", linestyle="--",
                   linewidth=2, label="Target deadline")

    ax.invert_yaxis()
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (days)")
    ax.set_title("Gantt Chart: original plan, replanned schedule, and actuals")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="lightgray", label="Original plan"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Replanned"),
        plt.Rectangle((0, 0), 1, 1, color="black", label="Actual"),
    ])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

