import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def exp_func(x, a, b, c):
    """Экспоненциальная аппроксимация"""
    return a * np.exp(b * x) + c
def hyp_func(x, a, b, c):
    """Экспоненциальная аппроксимация"""
    return a / (x - b) + c

def plot_idle_vs_duration(durations, idles_sum, percentiles_tasks, n_iter, save_path, max_duration=None):
    """
    Рисует график Парето: средняя длительность проекта vs средний суммарный простой.
    
    :param durations: список средних длительностей проекта
    :param idles_sum: список средних суммарных простоев
    :param percentiles_tasks: список процентилей задач
    :param n_iter: количество итераций Монте-Карло
    :param save_path: путь для сохранения картинки
    :param max_duration: если указан, рисуется вертикальная линия, а точки правее неё становятся серыми и прозрачными
    """
    plt.figure(figsize=(8, 6))

    np_durations = np.array(durations)
    np_idles_sum = np.array(idles_sum)
    percentiles_tasks = np.array(percentiles_tasks)

    # --- Если max_duration задан, разделяем точки ---
    if max_duration is not None:
        mask_right = np_durations > max_duration    # точки правее
        mask_left = ~mask_right                  # точки слева или на линии

        # обычные точки (левые)
        scatter_left = plt.scatter(
            np_durations[mask_left], np_idles_sum[mask_left],
            c=percentiles_tasks[mask_left], cmap='viridis',
            s=80, edgecolors='black'
        )

        # серые полупрозрачные точки (правые)
        scatter_right = plt.scatter(
            np_durations[mask_right], np_idles_sum[mask_right],
            color='gray', alpha=0.4,
            s=80, edgecolors='black'
        )

        # цветовая шкала — только для левых точек
        cbar = plt.colorbar(scatter_left)
        cbar.set_label("Процентиль задач")

        # вертикальная линия
        plt.axvline(max_duration, linestyle='--', color='red', linewidth=1.5)

    else:
        # обычный scatter
        scatter = plt.scatter(
            np_durations, np_idles_sum,
            c=percentiles_tasks, cmap='viridis',
            s=80, edgecolors='black'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Процентиль задач")

    # --- Экспоненциальная аппроксимация ---
    try:
        popt, _ = curve_fit(hyp_func, durations, idles_sum, p0=(1, 0.01, 1), maxfev=10000)
        print(f"⚙️ Параметры аппроксимации: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
        x_fit = np.linspace(min(np_durations), max(np_durations), 300)
        y_fit = hyp_func(x_fit, *popt)
        # plt.plot(x_fit, y_fit, "r--", linewidth=2, label="Экспоненциальная аппроксимация")
    except RuntimeError:
        print("⚠️ Не удалось выполнить экспоненциальную аппроксимацию")
    # --------------------------------------

    # Подписи точек
    for i, p in enumerate(percentiles_tasks):
        dx = 0.3
        dy = 0.3
        plt.text(np_durations[i] + dx, np_idles_sum[i] + dy, f"{p * 100:.0f}", fontsize=8)

    plt.xlabel("Средняя длительность проекта (дни)")
    plt.ylabel("Средний суммарный простой (дни)")
    plt.title(f"Pareto: простои vs длительность ({n_iter} итераций)")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"График сохранен в {save_path}")