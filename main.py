from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from services.metrics import calculate_project_duration, calculate_idle_time
from services.exporter import export_schedule_to_excel
from visualization.gantt_chart import plot_gantt

from services.metrics import monte_carlo_simulation
from services.exporter import export_percentile_analysis_to_excel
from visualization.plot_percentiles import plot_percentile_analysis
from visualization.plot_percentiles_ends_distr import plot_percentile_distributions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
import os
from scipy.stats import norm

def part1_1_schedule_project(with_buffer, path="data/tasks.csv", percentile=0.9, export_excel=True ):
    tasks = load_tasks_from_csv(path)
    scheduled_tasks = build_schedule(tasks, percentile, with_buffer=with_buffer, seed=None)
    project_duration = calculate_project_duration(scheduled_tasks)

    idle = calculate_idle_time(tasks)
    
    plot_gantt(scheduled_tasks, 'output/plots/gantt_with_buffer.png' if with_buffer else 'output/plots/gantt_without_buffer.png')

    if export_excel:
        export_schedule_to_excel(
            tasks,
            filename= "output/output_schedule_with_buffer.xlsx" if with_buffer else "output/output_schedule_without_buffer.xlsx",
            project_duration=project_duration,
            idle_time=idle
        )

def part1_2_explore_percentile_effect(percentiles, with_buffer, task_file="data/tasks.csv", n_iter=1_000, seed=None, output_excel="output/percentile_analysis.xlsx"):
    results = []

    for p in tqdm(percentiles, desc="Анализ процентилей"):
        # Плановое расписание для дедлайна
        base_tasks = load_tasks_from_csv(task_file)
        base_tasks = build_schedule(base_tasks, with_buffer=with_buffer, percentile=p, seed=seed)

        durations, idles = monte_carlo_simulation(task_file, p, n_iter, seed, with_buffer=with_buffer)

        plot_percentile_distributions(durations, p, f'output/plots/project_duration_distributions_{p}_with_buffer.png' if with_buffer else f'output/plots/project_duration_distributions_{p}_without_buffer.png')


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

        plt.savefig(f"output/plots/percentile_analysis_stacked_{p}_with_buffer.png" if with_buffer else f"output/plots/percentile_analysis_stacked_{p}_without_buffer.png", bbox_inches='tight')
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

    df = export_percentile_analysis_to_excel(results, output_excel)
    plot_percentile_analysis(df, "output/plots/percentile_analysis_with_buffer.png" if with_buffer else "output/plots/percentile_analysis_without_buffer.png")

if __name__ == "__main__":
    # with buffer
    part1_1_schedule_project(with_buffer=True, percentile=0.9)
    part1_2_explore_percentile_effect(with_buffer=True, percentiles=np.arange(0.5, 0.96, 0.05), output_excel="output/percentile_analysis_with_buffer.xlsx")
    
    #without buffer
    part1_1_schedule_project(with_buffer=False, percentile=0.9)
    part1_2_explore_percentile_effect(with_buffer=False, percentiles=np.arange(0.5, 0.96, 0.05), output_excel="output/percentile_analysis_without_buffer.xlsx")
