import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_gantt(tasks, filename, pr_buffer=None):
    """
    Старый универсальный вариант.
    """
    _, ax = plt.subplots(figsize=(14, 6))

    roles = sorted({task.role for task in tasks})
    palette = cm.get_cmap('Accent', max(len(roles), 1))
    color_map = {role: palette(i) for i, role in enumerate(roles)}

    y_labels = []
    yticks = []

    def sort_key(t):
        return (
            t.real_start_time if t.real_start_time is not None else 10**9,
            t.planned_start_time if t.planned_start_time is not None else 10**9,
            t.task_id
        )

    for i, task in enumerate(sorted(tasks, key=sort_key)):
        y_labels.append(f"Задача {task.task_id} ({task.role})")
        yticks.append(i)

        role_color = color_map[task.role]

        if task.planned_start_time is not None and task.planned_duration is not None:
            ax.barh(
                i, task.planned_duration,
                left=task.planned_start_time,
                height=0.5,
                color=role_color,
                alpha=0.75
            )

        if task.real_start_time is not None and task.real_duration is not None:
            ax.barh(
                i, task.real_duration,
                left=task.real_start_time,
                height=0.2,
                color='black'
            )

    if pr_buffer is not None and pr_buffer > 0:
        planned_end_project = max(
            [task.planned_end_time for task in tasks if task.planned_end_time is not None],
            default=0.0
        )
        ax.barh(
            len(tasks),
            pr_buffer,
            left=planned_end_project,
            height=0.5,
            color='red',
            alpha=0.6
        )
        y_labels.append("Буфер проекта")
        yticks.append(len(tasks))

    ax.invert_yaxis()
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Время")
    ax.set_title("Диаграмма Ганта")

    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color='black', label='Факт'),
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.6, label='План')
    ])

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_replanned_gantt(original_tasks, replanned_tasks, filename, target_finish_time=None):
    """
    Новый вариант:
      - серый: исходный план
      - цветной: новый перепланированный план
      - чёрный: реконструированный/известный факт
    """
    _, ax = plt.subplots(figsize=(16, 8))

    original_map = {t.task_id: t for t in original_tasks}
    roles = sorted({t.role for t in replanned_tasks})
    palette = cm.get_cmap('Accent', max(len(roles), 1))
    color_map = {role: palette(i) for i, role in enumerate(roles)}

    tasks_sorted = sorted(
        replanned_tasks,
        key=lambda t: (
            t.planned_start_time if t.planned_start_time is not None else 10**9,
            t.task_id
        )
    )

    y_labels = []
    yticks = []

    for i, task in enumerate(tasks_sorted):
        old_task = original_map[task.task_id]
        role_color = color_map[task.role]

        y_labels.append(f"Задача {task.task_id} ({task.role})")
        yticks.append(i)

        if old_task.planned_start_time is not None and old_task.planned_duration is not None:
            ax.barh(
                i,
                old_task.planned_duration,
                left=old_task.planned_start_time,
                height=0.55,
                color="lightgray",
                alpha=0.95
            )

        if task.planned_start_time is not None and task.planned_duration is not None:
            ax.barh(
                i,
                task.planned_duration,
                left=task.planned_start_time,
                height=0.35,
                color=role_color,
                alpha=0.85
            )

        if task.real_start_time is not None and task.real_duration is not None:
            ax.barh(
                i,
                task.real_duration,
                left=task.real_start_time,
                height=0.18,
                color="black",
                alpha=1.0
            )

    if target_finish_time is not None:
        ax.axvline(
            target_finish_time,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Целевой срок"
        )

    ax.invert_yaxis()
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Время")
    ax.set_title("Диаграмма Ганта: исходный план, факт и перепланирование")

    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="lightgray", label="Исходный план"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Новый план"),
        plt.Rectangle((0, 0), 1, 1, color="black", label="Факт"),
    ])

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()