import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_gantt(tasks, filename):
    _, ax = plt.subplots(figsize=(14, 6))

    roles = sorted({task.role for task in tasks})
    color_map = dict(zip(roles, cm.get_cmap('Accent').colors))

    y_labels = []
    yticks = []

    for i, task in enumerate(sorted(tasks, key=lambda t: t.start_time)):
        y_labels.append(f"Задача {task.task_id} ({task.role})")
        yticks.append(i)

        role_color = color_map[task.role]

        # Плановое время
        ax.barh(i, task.planned_duration, left=task.start_time, height=0.5, color=role_color, alpha=0.8)

        # Реальное выполнение (поверх плана, уже с другим уровнем)
        ax.barh(i, task.real_duration, left=task.start_time, height=0.2, color='black')

    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Время")
    ax.set_title("Диаграмма Ганта: план vs. факт (цвет = роль)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color='black', label='Факт'),
        plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.6, label='План')
    ])
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
