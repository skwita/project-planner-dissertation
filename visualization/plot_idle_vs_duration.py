import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def exp_func(x, a, b, c):
    """Экспоненциальная аппроксимация"""
    return a * np.exp(b * x) + c

def plot_idle_vs_duration(durations, idles_sum, percentiles_tasks, n_iter, save_path):
    """
    Рисует график Парето: средняя длительность проекта vs средний суммарный простой.
    
    :param durations: список средних длительностей проекта
    :param idles_sum: список средних суммарных простоев
    :param percentiles_tasks: список процентилей задач
    :param n_iter: количество итераций Монте-Карло
    :param save_path: путь для сохранения картинки
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        durations, idles_sum, 
        c=percentiles_tasks, cmap='viridis', 
        s=80, edgecolors='black'
    )
    
        
    # --- Экспоненциальная аппроксимация ---
    try:
        popt, _ = curve_fit(exp_func, durations, idles_sum, p0=(1, 0.01, 1), maxfev=10000)
        x_fit = np.linspace(min(durations), max(durations), 300)
        y_fit = exp_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, "r--", linewidth=2, label="Экспоненциальная аппроксимация")
    except RuntimeError:
        print("⚠️ Не удалось выполнить экспоненциальную аппроксимацию")
    # --------------------------------------
    

    # Подписи точек
    for i, p in enumerate(percentiles_tasks):
        plt.text(durations[i] + 0.3, idles_sum[i] + 0.3, f"{p:.2f}", fontsize=8)
    
    plt.xlabel("Средняя длительность проекта (дни)")
    plt.ylabel("Средний суммарный простой (дни)")
    plt.title(f"Pareto: простои vs длительность ({n_iter} итераций)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Процентиль задач")
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"График сохранен в {save_path}")