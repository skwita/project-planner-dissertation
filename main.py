from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from services.metrics import calculate_project_duration, calculate_idle_time, monte_carlo_simulation, calculate_buffer
from services.exporter import export_schedule_to_excel, export_percentile_analysis_to_excel
from visualization.gantt_chart import plot_gantt
from visualization.plot_percentiles_ends_distr import plot_percentile_pdf, plot_percentile_cdfs
from visualization.plot_idle_vs_duration import plot_idle_vs_duration

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def part1_1_schedule_project(pr_buffer, path="data/tasks.csv", percentile=0.9, export_excel=True ):
    tasks = load_tasks_from_csv(path)
    scheduled_tasks = build_schedule(tasks, percentile, seed=None)
    project_duration = calculate_project_duration(scheduled_tasks)

    idle = calculate_idle_time(tasks)
    
    plot_gantt(scheduled_tasks, f'output/plots/gantt_{percentile}.png', pr_buffer)

    if export_excel:
        export_schedule_to_excel(
            tasks,
            filename= "output/output_schedule.xlsx",
            project_duration=project_duration,
            idle_time=idle
        )

def part1_2_explore_percentile_effect(percentiles, task_file="data/tasks.csv", n_iter=1_000, seed=None):
    results = []
    all_durations = {}

    # собираем все данные
    for p in tqdm(percentiles, desc="Анализ процентилей"):
        base_tasks = load_tasks_from_csv(task_file)
        base_tasks = build_schedule(base_tasks, percentile=p, seed=seed)

        durations, idles = monte_carlo_simulation(task_file, p, n_iter, seed)
        all_durations[p] = durations

        avg_duration = np.mean(durations)
        roles = {role for idle in idles for role in idle}
        avg_idle = {role: np.mean([idle.get(role, 0.0) for idle in idles]) for role in roles}
        
        row = {"Процентиль": p, "Среднее время проекта": round(avg_duration, 2)}
        for role, idle in avg_idle.items():
            row[f"Простой_{role}"] = round(idle, 2)
        results.append(row)

    # общий диапазон по X (по квантилям, чтобы убрать хвосты)
    all_values = np.concatenate(list(all_durations.values()))
    # xmin, xmax = np.percentile(all_values, [1, 99])
    xmin, xmax = int(np.min(all_values) - 1), int(np.max(all_values) + 1)

    # === ОТДЕЛЬНЫЕ графики PDF и CDF для каждого процентиля ===
    for p, durations in all_durations.items():
        # PDF
        plot_percentile_pdf(
            durations_list=[durations],
            labels=[f"p={p:.2f}"],
            filename=f"output/plots/pdf_percentile_{p:.2f}.png",
            bins=xmax-xmin,
            xlim=(xmin, xmax)
        )
        # CDF
        plot_percentile_cdfs(
            durations_list=[durations],
            labels=[f"p={p:.2f}"],
            filename=f"output/plots/cdf_percentile_{p:.2f}.png",
            xlim=(xmin, xmax)
        )

    # экспорт в Excel
    df = export_percentile_analysis_to_excel(results, "output/percentile_analysis.xlsx")
    return df

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
    t_n = calculate_buffer(durations, calculate_project_duration(scheduled_tasks),  percentile_project)


    print(f"Плановая длительность: {planned_duration:.2f} дней")
    print(f"Буфер проекта ({percentile_project}%): {t_n :.2f} дней")

    return t_n

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
    plot_idle_vs_duration(durations, idles_sum, percentiles_tasks, n_iter, save_path)

def part1_5_multiple_percentiles(percentiles, task_file="data/tasks.csv", seed=None):
    res_dur = []
    for p in tqdm(percentiles, desc="Анализ процентилей"):
        # Плановое расписание для дедлайна
        base_tasks = load_tasks_from_csv(task_file)
        base_tasks = build_schedule(base_tasks, percentile=p, seed=seed)
        durations, _ = monte_carlo_simulation(task_file, p, 1000, seed)
        res_dur.append(durations)
    plot_percentile_pdf(res_dur, percentiles, 'output/plots/project_duration_distributions_multiple_percentiles.png')

def part1_6_plot_heatmaps(task_percentiles, project_percentiles):
    part1_6_1_heatmap_durations(task_percentiles=task_percentiles, project_percentiles=project_percentiles)
    part1_6_2_heatmap_human_resources(task_percentiles=task_percentiles, project_percentiles=project_percentiles)
    part1_6_3_heatmap_project_buffer(task_percentiles=task_percentiles, project_percentiles=project_percentiles)

def part1_6_1_heatmap_durations(task_file="data/tasks.csv", 
                                task_percentiles=[0.5, 0.7, 0.9], 
                                project_percentiles=[0.5, 0.7, 0.9], 
                                seed=None, n_sim=100):
    """
    Тепловая карта длительности проекта в зависимости от процентиля задачи и проектного процентиля.
    """
    durations_matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    for i, t_p in enumerate(tqdm(task_percentiles, desc="Task percentiles")):
        for j, p_p in enumerate(project_percentiles):
            # Считаем длительность проекта через Монте-Карло
            sim_durations, _ = monte_carlo_simulation(task_file, t_p, n_sim, seed=seed)
            pr_buffer = part1_3_project_buffer(percentile_tasks=t_p, percentile_project=p_p * 100) #TODO
            durations_matrix[i, j] = np.average(sim_durations) + pr_buffer #TODO

    plt.figure(figsize=(10, 6))
    sns.heatmap(durations_matrix, 
                annot=True, fmt=".1f", cmap="mako", 
                xticklabels=[f"p={p :.2f}" for p in project_percentiles],
                yticklabels=[f"p={t :.2f}" for t in task_percentiles])

    plt.xlabel("Процентиль проекта")
    plt.ylabel("Процентиль задачи")
    plt.title("Тепловая карта длительности проекта")
    plt.tight_layout()
    plt.savefig("output/plots/heatmap_durations_with_buffer.png", dpi=300) #TODO
    plt.close()

def part1_6_2_heatmap_human_resources(task_file="data/tasks.csv", 
                                task_percentiles=[0.5, 0.7, 0.9], 
                                project_percentiles=[0.5, 0.7, 0.9], 
                                seed=None, n_sim=100):
    """
    Тепловая карта трудовых ресурсов проекта в зависимости от процентиля задачи и проектного процентиля.
    """
    durations_matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    for i, t_p in enumerate(tqdm(task_percentiles, desc="Task percentiles")):
        for j, p_p in enumerate(project_percentiles):
            # Считаем длительность проекта через Монте-Карло
            _, idles = monte_carlo_simulation(task_file, t_p, n_sim, seed=seed)

            ammount = 1
            summ = 0
            for idle in idles:
                summ += sum(idle.values())
                ammount += 1

            durations_matrix[i, j] = summ / ammount

    plt.figure(figsize=(10, 6))
    sns.heatmap(durations_matrix, 
                annot=True, fmt=".1f", cmap="mako", 
                xticklabels=[f"p={p :.2f}" for p in project_percentiles],
                yticklabels=[f"p={t :.2f}" for t in task_percentiles])

    plt.xlabel("Процентиль проекта")
    plt.ylabel("Процентиль задачи")
    plt.title("Тепловая карта трудовых ресурсов проекта")
    plt.tight_layout()
    plt.savefig("output/plots/heatmap_hr.png", dpi=300)
    plt.close()

def part1_6_3_heatmap_project_buffer(task_file="data/tasks.csv", 
                                task_percentiles=[0.5, 0.7, 0.9], 
                                project_percentiles=[0.5, 0.7, 0.9], 
                                seed=None, n_sim=100):
    """
    Тепловая карта буфера проекта в зависимости от процентиля задачи и проектного процентиля.
    """
    durations_matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    for i, t_p in enumerate(tqdm(task_percentiles, desc="Task percentiles")):
        for j, p_p in enumerate(project_percentiles):
            # Считаем длительность проекта через Монте-Карло
            pr_buffer = part1_3_project_buffer(percentile_tasks=t_p, percentile_project=p_p * 100)
            durations_matrix[i, j] = pr_buffer

    plt.figure(figsize=(10, 6))
    sns.heatmap(durations_matrix, 
                annot=True, fmt=".1f", cmap="mako", 
                xticklabels=[f"p={p :.2f}" for p in project_percentiles],
                yticklabels=[f"p={t :.2f}" for t in task_percentiles])

    plt.xlabel("Процентиль проекта")
    plt.ylabel("Процентиль задачи")
    plt.title("Тепловая карта буферов проекта")
    plt.tight_layout()
    plt.savefig("output/plots/heatmap_buffer.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    PERCENTILE_TASK = 0.5
    PERCENTILE_PROJECT = 0.9
    PERCENTILES_RANGE = np.arange(0.05, 0.96, 0.05)
    PERCENTILES_FOR_PLOT = [0.3, 0.6, 0.9]

    # Нахождение буфера проекта
    pr_buffer = part1_3_project_buffer(percentile_tasks=PERCENTILE_TASK, percentile_project=PERCENTILE_PROJECT * 100)
    # Расчет задач, построение диаграммы Гантта, экспорт таблицы задач
    part1_1_schedule_project(pr_buffer, percentile=PERCENTILE_TASK)
    # # Построение графиков кумулятивных функций распределения и плотности вероятности
    # part1_2_explore_percentile_effect(percentiles=PERCENTILES_RANGE)
    # # Построение Парето графика (Простои-Длительность для разных процентилей задач)
    # part1_4_plot_pareto_idle_vs_duration(PERCENTILES_RANGE)
    # # Построение графика плотности вероятности с разными процентилями
    # part1_5_multiple_percentiles(PERCENTILES_FOR_PLOT)
    # # Тепловые карты по длительности, 
    # part1_6_plot_heatmaps(task_percentiles=PERCENTILES_RANGE, project_percentiles=PERCENTILES_RANGE)
   
    # Нахождение буфера проекта
    pr_buffer = part1_3_project_buffer(percentile_tasks=0.3, percentile_project=PERCENTILE_PROJECT * 100)
    # Расчет задач, построение диаграммы Гантта, экспорт таблицы задач
    part1_1_schedule_project(pr_buffer, percentile=0.3)
    # Нахождение буфера проекта
    pr_buffer = part1_3_project_buffer(percentile_tasks=0.9, percentile_project=PERCENTILE_PROJECT * 100)
    # Расчет задач, построение диаграммы Гантта, экспорт таблицы задач
    part1_1_schedule_project(pr_buffer, percentile=0.9)