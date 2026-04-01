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

def plot_pareto_transition(
    durations_base,
    idles_base,
    durations_new,
    idles_new,
    percentiles,
    deadline,
    save_path
):
    """
    Визуализация перехода решения на Парето-кривой:
    - исходная кривая
    - новая кривая после смещения
    - стрелка перехода решения
    """

    plt.figure(figsize=(9, 7))

    # --- массивы ---
    d0 = np.array(durations_base)
    i0 = np.array(idles_base)

    d1 = np.array(durations_new)
    i1 = np.array(idles_new)

    percentiles = np.array(percentiles)

    # --- 1. исходная кривая ---
    plt.scatter(d0, i0, c=percentiles, cmap='Blues',
                s=70, edgecolors='black', label="Исходная")

    # --- 2. новая кривая ---
    plt.scatter(d1, i1, c=percentiles, cmap='Greens',
                s=70, edgecolors='black', label="После переоценки")

    # --- линия дедлайна ---
    plt.axvline(deadline, linestyle='--', color='red', linewidth=2, label="Дедлайн")

    # --- 3. исходное решение (ближайшее к дедлайну слева) ---
    mask0 = d0 <= deadline
    idx0 = np.argmax(d0[mask0])
    x0 = d0[mask0][idx0]
    y0 = i0[mask0][idx0]

    # --- 4. новое решение ---
    mask1 = d1 <= deadline
    idx1 = np.argmax(d1[mask1])
    x1 = d1[mask1][idx1]
    y1 = i1[mask1][idx1]

    # --- выделение точек ---
    plt.scatter(x0, y0, color='blue', s=140, zorder=5)
    plt.scatter(x1, y1, color='green', s=140, zorder=5)

    # --- стрелка перехода ---
    plt.arrow(
        x0, y0,
        x1 - x0, y1 - y0,
        head_width=0.5,
        length_includes_head=True,
        color='black',
        linewidth=2
    )

    # --- подписи ---
    plt.text(x0, y0, "старый план", fontsize=10, color='blue')
    plt.text(x1, y1, "новый план", fontsize=10, color='green')

    # --- подписи процентилей ---
    for i, p in enumerate(percentiles):
        plt.text(d1[i] + 0.3, i1[i] + 0.3, f"{p:.2f}", fontsize=7)

    plt.xlabel("Длительность проекта")
    plt.ylabel("Суммарный простой")
    plt.title("Переход решения на Парето-фронте при переоценке")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"График перехода сохранен в {save_path}")

def plot_history_metrics(history, save_prefix="output/plots/history"):
    import matplotlib.pyplot as plt

    stages = [h["stage"] for h in history]
    bias = [h["bias"] for h in history]
    percentile = [h["percentile"] for h in history]
    finish = [h["finish"] for h in history]

    # --- 1. Смещение ---
    plt.figure()
    plt.plot(stages, bias, marker='o')
    plt.xlabel("Этап")
    plt.ylabel("Смещение (bias)")
    plt.title("Динамика смещения")
    plt.grid(True)
    plt.savefig(f"{save_prefix}_bias.png", dpi=300)
    plt.close()

    # --- 2. Percentile ---
    plt.figure()
    plt.plot(stages, percentile, marker='o')
    plt.xlabel("Этап")
    plt.ylabel("Percentile")
    plt.title("Динамика процентиля")
    plt.grid(True)
    plt.savefig(f"{save_prefix}_percentile.png", dpi=300)
    plt.close()

    # --- 3. Срок проекта ---
    plt.figure()
    plt.plot(stages, finish, marker='o')
    plt.xlabel("Этап")
    plt.ylabel("Срок проекта")
    plt.title("Динамика срока проекта")
    plt.grid(True)
    plt.savefig(f"{save_prefix}_finish.png", dpi=300)
    plt.close()

    print("Графики сохранены")