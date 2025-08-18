import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patheffects as pe

def plot_percentile_pdf(durations_list, labels, filename, bins=40, xlim=(40, 80)):
    """
    Рисует распределения длительности проекта для нескольких наборов данных:
    - гистограмма (фон)
    - KDE-кривая поверх неё

    :param durations_list: список массивов с длительностями (например, [dur1, dur2, dur3])
    :param labels: список подписей (например, ["p=50%", "p=70%", "p=90%"])
    :param filename: путь для сохранения графика
    :param bins: количество корзин гистограммы
    :param xlim: диапазон по оси X (tuple)
    """
    plt.figure(figsize=(12, 8))

    colors = ["skyblue", "orange", "green", "red", "purple"]

    x_grid = np.linspace(xlim[0], xlim[1], 1000)

    for i, durations in enumerate(durations_list):
        color = colors[i % len(colors)]

        # гистограмма
        plt.hist(
            durations, bins=bins, range=xlim, density=True,
            alpha=0.4, color=color, edgecolor="black", label=f"{labels[i]} (гист.)"
        )

        # KDE
        kde = gaussian_kde(durations)
        line, = plt.plot(x_grid, kde(x_grid), color=color, linewidth=2.5, label=labels[i])
        line.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])

    plt.xlabel("Длительность проекта (дни)")
    plt.ylabel("Плотность вероятности")
    plt.title("Распределение длительности проекта (разные процентили задач)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_percentile_cdfs(durations_list, labels, filename, xlim=None):
    """
    Рисует CDF (накопленные распределения) для нескольких наборов данных.

    :param durations_list: список массивов с длительностями (например, [dur1, dur2, dur3])
    :param labels: список подписей (например, ["p=50%", "p=70%", "p=90%"])
    :param filename: путь для сохранения графика
    :param xlim: диапазон по оси X (tuple или None)
    """
    plt.figure(figsize=(12, 8))

    colors = ["skyblue", "orange", "green", "red", "purple"]

    for i, durations in enumerate(durations_list):
        durations = np.sort(durations)
        n = len(durations)
        cdf_vals = np.arange(1, n + 1) / n
        color = colors[i % len(colors)]
        plt.plot(durations, cdf_vals, color=color, linewidth=2.5, label=labels[i])

    if xlim:
        plt.xlim(xlim)

    plt.xlabel("Длительность проекта (дни)")
    plt.ylabel("Накопленная вероятность (CDF)")
    plt.title("CDF длительности проекта (разные процентили задач)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()