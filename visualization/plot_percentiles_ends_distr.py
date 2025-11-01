import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.patheffects as pe

def plot_percentile_pdf(durations_list, labels, filename,
                        bins=40, xlim=(40, 80)):
    plt.figure(figsize=(12, 8))
    colors = ["skyblue", "orange", "green", "red", "purple"]

    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    cutoff = bin_edges[2]  # правая граница второго бина (исключаем первые два столбика)
    x_grid = np.linspace(cutoff, xlim[1], 1000)  # начинаем сетку от cutoff

    for i, durations in enumerate(durations_list):
        durations = np.asarray(durations)
        color = colors[i % len(colors)]

        # гистограмма по всем данным
        plt.hist(durations, bins=bins, range=xlim, density=True,
                 alpha=0.4, color=color, edgecolor="black",
                 label=f"{labels[i]} (гист.)")

        # отфильтровываем точки из первого и второго бина
        filtered = durations[durations >= cutoff]

        if filtered.size > 1:
            kde = gaussian_kde(durations)
            y = kde(x_grid)

            # нормализуем не к 1, а к вероятности "массы" правее cutoff
            mass = np.sum((durations >= cutoff)) / len(durations)
            area = np.trapezoid(y, x_grid)
            if area > 0:
                y = y * mass / area

            line, = plt.plot(x_grid, y, color=color, linewidth=2.5, label=labels[i])
            line.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
        else:
            print(f"⚠️ Для {labels[i]} слишком мало точек после удаления первых двух бинов.")

    plt.xlabel("Длительность проекта (дни)")
    plt.ylabel("Плотность вероятности")
    plt.title("Распределение длительности проекта (первые два бина исключены из KDE)")
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
        plt.plot(durations, cdf_vals, color=color, linewidth=2.5, label=labels[i] * 100)

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