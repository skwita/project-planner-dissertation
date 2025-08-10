from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from services.metrics import calculate_project_duration, calculate_idle_time, monte_carlo_simulation
from services.exporter import export_schedule_to_excel, export_percentile_analysis_to_excel
from visualization.gantt_chart import plot_gantt
from visualization.plot_percentiles import plot_percentile_analysis
from visualization.plot_percentiles_ends_distr import plot_percentile_distributions

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def part1_1_schedule_project(path="data/tasks.csv", percentile=0.9, export_excel=True ):
    tasks = load_tasks_from_csv(path)
    scheduled_tasks = build_schedule(tasks, percentile, seed=None)
    project_duration = calculate_project_duration(scheduled_tasks)

    idle = calculate_idle_time(tasks)
    
    plot_gantt(scheduled_tasks, 'output/plots/gantt.png')

    if export_excel:
        export_schedule_to_excel(
            tasks,
            filename= "output/output_schedule.xlsx",
            project_duration=project_duration,
            idle_time=idle
        )

def part1_2_explore_percentile_effect(percentiles, task_file="data/tasks.csv", n_iter=1_000, seed=None):
    results = []

    for p in tqdm(percentiles, desc="Анализ процентилей"):
        # Плановое расписание для дедлайна
        base_tasks = load_tasks_from_csv(task_file)
        base_tasks = build_schedule(base_tasks, percentile=p, seed=seed)

        durations, idles = monte_carlo_simulation(task_file, p, n_iter, seed)

        plot_percentile_distributions(durations, p, f'output/plots/project_duration_distributions_{p}.png')
###
        sorted_durations = sorted(durations)
        # Создаем массивы для значений и их вероятностей
        # x_values = np.unique(sorted_durations)
        x_values = sorted_durations
        y_prob = []

        n = len(durations)
        for x in x_values:
            prob = np.sum(np.array(sorted_durations) <= x) / n
            y_prob.append(prob)

        
        plt.plot(x_values, y_prob)
        plt.xlim(0,200)

        plt.savefig(f"output/plots/percentile_analysis_stacked_{p}.png", bbox_inches='tight')
        plt.close()
###
        avg_duration = np.mean(durations)
        roles = {role for idle in idles for role in idle}
        avg_idle = {role: np.mean([idle.get(role, 0.0) for idle in idles]) for role in roles}
        
        row = {
            "Процентиль": p,
            "Среднее время проекта": round(avg_duration, 2)
        }

        for role, idle in avg_idle.items():
            row[f"Простой_{role}"] = round(idle, 2)

        results.append(row)

    df = export_percentile_analysis_to_excel(results, "output/percentile_analysis.xlsx")
    plot_percentile_analysis(df, "output/plots/percentile_analysis.png")

def part1_3_project_buffer(percentile_tasks=0.5, percentile_project=0.9, task_file="data/tasks.csv", n_iter=1000, seed=None):
    """
    Рассчитывает размер буфера проекта (buffer_90) как:
    buffer_90 = t90 - плановое время окончания последней задачи.
    
    :param percentile_tasks: процентиль длительностей задач для планирования
    :param task_file: путь к CSV с задачами
    :param n_iter: количество симуляций
    :param seed: фиксированное зерно генератора
    :return: размер буфера в днях
    """
    # 1. Плановое расписание
    tasks = load_tasks_from_csv(task_file)
    scheduled_tasks = build_schedule(tasks, percentile=percentile_tasks, seed=seed)
    planned_duration = max(task.planned_end_time for task in scheduled_tasks)

    # 2. Моделирование N проектов
    durations, _ = monte_carlo_simulation(task_file, percentile_tasks, n_iter, seed)

    # 3. t90 — длительность, в которую укладывается 90% проектов
    t_n = np.percentile(durations, percentile_project)

    # 4. Расчет буфера
    buffer_n = t_n - planned_duration

    print(f"Плановая длительность: {planned_duration:.2f} дней")
    print(f"Длительность t{percentile_project*100} проекта: {t_n:.2f} дней")
    print(f"Буфер проекта ({percentile_project*100}%): {buffer_n:.2f} дней")

    return buffer_n

def part1_4_plot_pareto_idle_vs_duration(percentiles_tasks, task_file="data/tasks.csv", seed=None, n_iter=1000, save_path="output/plots/pareto_idle_duration.png"):
    """
    Строит график Парето: средняя длительность проекта vs средний суммарный простой
    при разных перцентилях задач, рассчитанные по результатам Monte Carlo.

    :param percentiles_tasks: список процентилей для длительностей задач (0.1 = 10%)
    :param task_file: путь к CSV с задачами
    :param seed: зерно генератора для воспроизводимости
    :param n_iter: количество итераций Монте-Карло
    :param save_path: путь для сохранения графика
    """
    # ensure_dir(os.path.dirname(save_path))
    
    durations = []
    idles_sum = []
    
    for p in percentiles_tasks:
        # Прогоняем Монте-Карло
        mc_durations, mc_idles = monte_carlo_simulation(task_file, p, n_iter, seed)
        
        # Средняя длительность
        avg_duration = np.mean(mc_durations)
        
        # Средний суммарный простой
        avg_idle_sum = np.mean([sum(idle.values()) for idle in mc_idles])
        
        durations.append(avg_duration)
        idles_sum.append(avg_idle_sum)
    
    # Построение графика
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(durations, idles_sum, c=percentiles_tasks, cmap='viridis', s=80, edgecolors='black')
    
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

if __name__ == "__main__":
    # part1_1_schedule_project(percentile=0.5)
    # part1_2_explore_percentile_effect(percentiles=np.arange(0.05, 0.96, 0.05))
    # part1_3_project_buffer(percentile_tasks=0.1, percentile_project=0.9)
    part1_4_plot_pareto_idle_vs_duration(np.arange(0.05, 0.96, 0.05))
   