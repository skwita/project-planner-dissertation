from collections import defaultdict, deque
from concurrent import futures
import copy
from services.critical_path import critical_chain_path, find_critical_path
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from services.metrics import calculate_project_duration, calculate_idle_time, monte_carlo_schedules, monte_carlo_simulation, calculate_buffer, parallel_monte_carlo_simulation
from services.exporter import export_schedule_to_excel, export_percentile_analysis_to_excel
from visualization.gantt_chart import plot_gantt
from visualization.plot_percentiles_ends_distr import plot_percentile_pdf, plot_percentile_cdfs
from visualization.plot_idle_vs_duration import plot_idle_vs_duration
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import networkx as nx

matplotlib.use("Agg")

def part1_1_schedule_project(pr_buffer, path="data/tasks.csv", percentile=0.9, export_excel=True, runs=10000):
    print("______________________________________________________")
    print(f"part1_1 started at {datetime.now().time()}")
    print("______________________________________________________")

    tasks = load_tasks_from_csv(path)

    aggregated = defaultdict(lambda: defaultdict(list))

    for run in range(runs):
        tasks_copy = copy.deepcopy(tasks)
        scheduled = build_schedule(tasks_copy, percentile, seed=run)

        for task in scheduled:
            aggregated[task.task_id]["planned_start_time"].append(task.planned_start_time)
            aggregated[task.task_id]["planned_end_time"].append(task.planned_end_time)
            aggregated[task.task_id]["real_start_time"].append(task.real_start_time)
            aggregated[task.task_id]["real_end_time"].append(task.real_end_time)

    scheduled_tasks = copy.deepcopy(tasks)
    for task in scheduled_tasks:
        task.planned_start_time = float(np.mean(aggregated[task.task_id]["planned_start_time"]))
        task.planned_end_time   = float(np.mean(aggregated[task.task_id]["planned_end_time"]))
        task.real_start_time    = float(np.mean(aggregated[task.task_id]["real_start_time"]))
        task.real_end_time      = float(np.mean(aggregated[task.task_id]["real_end_time"]))

        # пересчёт длительностей
        task.planned_duration = task.planned_end_time - task.planned_start_time
        task.real_duration    = task.real_end_time - task.real_start_time

    # считаем метрики уже на усреднённых задачах
    project_duration = calculate_project_duration(scheduled_tasks)
    idle = calculate_idle_time(scheduled_tasks)

    plot_gantt(scheduled_tasks, f'output/plots/gantt_{percentile}.png', pr_buffer)

    if export_excel:
        export_schedule_to_excel(
            scheduled_tasks,
            filename="output/output_schedule.xlsx",
            project_duration=project_duration,
            idle_time=idle
        )

    return scheduled_tasks, project_duration, idle

def part1_2_explore_percentile_effect(percentiles, task_file="data/tasks.csv", n_iter=100_000, seed=None):
    print("______________________________________________________")
    print(f"part1_2 started at {datetime.now().time()}")
    print("______________________________________________________")
    results = []
    all_durations = {}
    
    # параллельный запуск всех симуляций
    parallel_results = parallel_monte_carlo_simulation(task_file, percentiles, n_iter, seed)

    # собираем все данные
    for p in percentiles:
        durations, idles = parallel_results[p]
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

def part1_3_project_buffer(percentile_tasks=0.5, percentile_project=0.9, task_file="data/tasks.csv", n_iter=1_000, seed=None):
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
    # planned_duration = max(task.planned_end_time for task in scheduled_tasks)

    # 2. Моделирование N проектов
    durations, _ = monte_carlo_simulation(task_file, percentile_tasks, n_iter, seed)

    # 3. t90 — длительность, в которую укладывается 90% проектов
    t_n = calculate_buffer(durations, calculate_project_duration(scheduled_tasks),  percentile_project)

    return t_n

def part1_4_plot_pareto_idle_vs_duration(percentiles_tasks, task_file="data/tasks.csv", seed=None, n_iter=100_000, save_path="output/plots/pareto_idle_duration.png"):
    """
    Строит график Парето: средняя длительность проекта vs средний суммарный простой
    при разных перцентилях задач, рассчитанные по результатам Monte Carlo.

    :param percentiles_tasks: список процентилей для длительностей задач (0.1 = 10%)
    :param task_file: путь к CSV с задачами
    :param seed: зерно генератора для воспроизводимости
    :param n_iter: количество итераций Монте-Карло
    :param save_path: путь для сохранения графика
    """

    print("______________________________________________________")
    print(f"part1_4 started at {datetime.now().time()}")
    print("______________________________________________________")

    durations = []
    idles_sum = []
    
    parallel_results = parallel_monte_carlo_simulation(task_file, percentiles_tasks, n_iter, seed)

    for p in percentiles_tasks:
        # Прогоняем Монте-Карло
        mc_durations, mc_idles = parallel_results[p]
        
        # Средняя длительность
        avg_duration = np.mean(mc_durations)
        
        # Средний суммарный простой
        avg_idle_sum = np.mean([sum(idle.values()) for idle in mc_idles])
        
        durations.append(avg_duration)
        idles_sum.append(avg_idle_sum)
    
    # Построение графика
    plot_idle_vs_duration(durations, idles_sum, percentiles_tasks, n_iter, save_path)

def part1_5_multiple_percentiles(percentiles, task_file="data/tasks.csv", seed=None):
    print("______________________________________________________")
    print(f"part1_5 started at {datetime.now().time()}")
    print("______________________________________________________")

    res_dur = []
    parallel_results = parallel_monte_carlo_simulation(task_file, percentiles, 100_000, seed)
    for p in percentiles:
        durations, _ = parallel_results[p]
        res_dur.append(durations)
    plot_percentile_pdf(res_dur, percentiles, 'output/plots/project_duration_distributions_multiple_percentiles_pdf.png')
    plot_percentile_cdfs(res_dur, percentiles, 'output/plots/project_duration_distributions_multiple_percentiles_cdf.png')

def part1_6_plot_heatmaps(task_percentiles, project_percentiles):
    print("______________________________________________________")
    print(f"part1_6 started at {datetime.now().time()}")
    print("______________________________________________________")
    part1_6_1_heatmap_durations(task_percentiles=task_percentiles, project_percentiles=project_percentiles)
    part1_6_2_heatmap_idles(task_percentiles=task_percentiles, project_percentiles=project_percentiles)
    part1_6_3_heatmap_project_buffer(task_percentiles=task_percentiles, project_percentiles=project_percentiles)

def compute_duration_and_buffer(task_file, t_p, p_p, n_sim, seed):
    sim_durations, _ = monte_carlo_simulation(task_file, t_p, n_sim, seed)
    pr_buffer = part1_3_project_buffer(t_p, p_p * 100)
    avg_duration = np.mean(sim_durations)
    return avg_duration, pr_buffer

def part1_6_1_heatmap_durations(task_file="data/tasks.csv", 
                                task_percentiles=[0.5, 0.7, 0.9], 
                                project_percentiles=[0.5, 0.7, 0.9], 
                                seed=None, n_sim=100):
    """
    Тепловая карта длительности проекта в зависимости от процентиля задачи и проектного процентиля.
    """
    durations_matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    future_to_idx = {}
    with futures.ProcessPoolExecutor() as executor:
        for i, t_p in enumerate(task_percentiles):
            for j, p_p in enumerate(project_percentiles):
                future = executor.submit(
                    compute_duration_and_buffer,
                    task_file, t_p, p_p, n_sim, seed
                )
                future_to_idx[future] = (i, j)

        for future in tqdm(futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Calculating durations"):
            i, j = future_to_idx[future]
            avg_duration, pr_buffer = future.result()
            durations_matrix[i, j] = avg_duration + pr_buffer

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(durations_matrix, 
                annot=True, fmt=".1f", cmap="mako", 
                xticklabels=[f"p={p :.2f}" for p in project_percentiles],
                yticklabels=[f"p={t :.2f}" for t in task_percentiles])
    ax.invert_yaxis()  # переворачиваем ось Y, чтобы 0-й индекс был снизу
    plt.xlabel("Процентиль проекта")
    plt.ylabel("Процентиль задачи")
    plt.title("Тепловая карта длительности проекта")
    plt.tight_layout()
    plt.savefig("output/plots/heatmap_durations_with_buffer.png", dpi=300) #TODO
    plt.close()

def compute_avg_idle(task_file, t_p, n_sim, seed):
    _, idles = monte_carlo_simulation(task_file, t_p, n_sim, seed)
    ammount = 1
    summ = 0
    for idle in idles:
        summ += sum(idle.values())
        ammount += 1
    avg_idle_sum = summ / ammount
    return avg_idle_sum

def part1_6_2_heatmap_idles(task_file="data/tasks.csv", 
                                task_percentiles=[0.5, 0.7, 0.9], 
                                project_percentiles=[0.5, 0.7, 0.9], 
                                seed=None, n_sim=100):
    """
    Тепловая карта трудовых ресурсов проекта в зависимости от процентиля задачи и проектного процентиля.
    """
    durations_matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    future_to_idx = {}
    with futures.ProcessPoolExecutor() as executor:
        for i, t_p in enumerate(task_percentiles):
            for j, p_p in enumerate(project_percentiles):
                future = executor.submit(
                    compute_avg_idle,
                    task_file, t_p, n_sim, seed
                )
                future_to_idx[future] = (i, j)

        for future in tqdm(futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Calculating idles"):
            i, j = future_to_idx[future]
            avg_idle_sum = future.result()
            durations_matrix[i, j] = avg_idle_sum

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(durations_matrix, 
                annot=True, fmt=".1f", cmap="mako", 
                xticklabels=[f"p={p :.2f}" for p in project_percentiles],
                yticklabels=[f"p={t :.2f}" for t in task_percentiles])
    ax.invert_yaxis()  # переворачиваем ось Y, чтобы 0-й индекс был снизу
    plt.xlabel("Процентиль проекта")
    plt.ylabel("Процентиль задачи")
    plt.title("Тепловая карта простоев")
    plt.tight_layout()
    plt.savefig("output/plots/heatmap_idle.png", dpi=300)
    plt.close()

def part1_6_3_heatmap_project_buffer(task_file="data/tasks.csv", 
                                task_percentiles=[0.5, 0.7, 0.9], 
                                project_percentiles=[0.5, 0.7, 0.9], 
                                seed=None, n_sim=100):
    """
    Тепловая карта буфера проекта в зависимости от процентиля задачи и проектного процентиля.
    """
    durations_matrix = np.zeros((len(task_percentiles), len(project_percentiles)))

    with futures.ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(part1_3_project_buffer, percentile_tasks=t_p, percentile_project=p_p * 100): (i, j)
            for i, t_p in enumerate(task_percentiles)
            for j, p_p in enumerate(project_percentiles)
        }

        for future in tqdm(futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Buffers"):
            i, j = future_to_idx[future]
            try:
                durations_matrix[i, j] = future.result()
            except Exception as e:
                durations_matrix[i, j] = np.nan
                print(f"Ошибка при расчете буфера для ({i}, {j}): {e}")

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(durations_matrix, 
                annot=True, fmt=".1f", cmap="mako", 
                xticklabels=[f"p={p :.2f}" for p in project_percentiles],
                yticklabels=[f"p={t :.2f}" for t in task_percentiles])
    ax.invert_yaxis()  # переворачиваем ось Y, чтобы 0-й индекс был снизу
    plt.xlabel("Процентиль проекта")
    plt.ylabel("Процентиль задачи")
    plt.title("Тепловая карта буферов проекта")
    plt.tight_layout()
    plt.savefig("output/plots/heatmap_buffer.png", dpi=300)
    plt.close()

def find_critical_path(tasks):
    """
    Находит критический путь среди задач Task.
    Возвращает список task_id в порядке выполнения.
    """
    G = nx.DiGraph()

    # Добавляем вершины и длительности
    for t in tasks:
        if t.planned_duration is None:
            if t.planned_start_time is not None and t.planned_end_time is not None:
                duration = t.planned_end_time - t.planned_start_time
            else:
                duration = t.mean  # fallback
        else:
            duration = t.planned_duration
        G.add_node(t.task_id, duration=duration)

    # Добавляем зависимости
    for t in tasks:
        for dep in t.dependencies:
            G.add_edge(dep, t.task_id)

    # Находим критический путь
    critical_path = nx.dag_longest_path(G, weight='duration')
    return critical_path


def buffer_penetration_timeline(tasks, project_buffer):
    """
    Классический график сгорания буфера (Buffer Fever Chart).
    X — прогресс проекта (% выполнения),
    Y — остаток буфера (% неиспользованного).
    """

    critical_tasks_ids = critical_chain_path(tasks)

    # crit_tasks = []
    # for t_id in critical_tasks_ids:
    #     thisTask = None
    #     for task in tasks:
    #         if task.task_id == t_id:
    #             thisTask = task
    #     crit_tasks.append(thisTask)
    crit_tasks = critical_tasks_ids
    crit_tasks.sort(key=lambda t: t.planned_end_time)

    # 3️⃣ Моделируем сгорание буфера
    total_delay = 0.0
    buffer_remaining = []
    times = []
    delays = []

    for i, task in enumerate(crit_tasks):
        if i != 0:
            delays.append(max(0, task.real_end_time - task.planned_end_time) - sum(delays[:i]))
        else:
            delays.append(max(0, task.real_end_time - task.planned_end_time))

    buffer_remaining.append(project_buffer)
    times.append(0)
    
    for i, t in enumerate(crit_tasks):
        plan_end = t.planned_end_time
        real_end = t.real_end_time

        # Проверка на корректность
        if plan_end is None or real_end is None:
            continue

        # Задержка текущей задачи
        # delay = max(0, real_end - plan_end)


        # # Остаток буфера
        # remaining = max(0, project_buffer - total_delay)
        # buffer_remaining.append(remaining / project_buffer)
        # times.append(plan_end)

        buffer_remaining.append(project_buffer)
        times.append(plan_end)
        # buffer_remaining.append(project_buffer - delays[i])
        # times.append(real_end)

        project_buffer -= delays[i]

    # 4️⃣ Визуализация
    plt.figure(figsize=(10, 5))

    # === 1. Фиксируем верхнюю точку графика ===
    initial_buffer = buffer_remaining[0]

    # === 2. Нормированный прогресс ===
    x0 = times[0]
    x1 = times[-1]
    progress = [(t - x0) / (x1 - x0) for t in times]

    # === 3. Центральная линия (идеальное сгорание) ===
    ideal_y = [initial_buffer * (1 - p) for p in progress]

    # === 4. Сигмы ===
    sigma = initial_buffer / 6  # чтобы ±3σ покрывали высоту буфера

    # === 5. Симметричный веер зон ===
    green_upper = [initial_buffer * (1 - p) + sigma * p for p in progress]
    green_lower = [initial_buffer * (1 - p) - sigma * p for p in progress]

    yellow_upper = [initial_buffer * (1 - p) + 2*sigma * p for p in progress]
    yellow_lower = [initial_buffer * (1 - p) - 2*sigma * p for p in progress]

    red_upper = [initial_buffer * (1 - p) + 3*sigma * p for p in progress]
    red_lower = [initial_buffer * (1 - p) - 3*sigma * p for p in progress]

    # === 6. Обрезка значений ===
    # def clamp(v): return max(0, v)
    # green_upper = [clamp(v) for v in green_upper]
    # yellow_upper = [clamp(v) for v in yellow_upper]
    # red_upper = [clamp(v) for v in red_upper]

    # green_lower = [clamp(v) for v in green_lower]
    # yellow_lower = [clamp(v) for v in yellow_lower]
    # red_lower = [clamp(v) for v in red_lower]


    # --- Окружение границ графика ---
    y_max = max(buffer_remaining) * 1.1  # или просто initial_buffer * 1.1
    y_min = 0

    # === Красная зона сверху ===
    plt.fill_between(
        times,
        y_max,
        yellow_upper,
        color="green",
        alpha=0.08
    )

    # === Красная зона снизу ===
    plt.fill_between(
        times,
        yellow_lower,
        min(buffer_remaining),
        color="red",
        alpha=0.08
    )

    # === 7. Отрисовка зон ===
    # plt.fill_between(times, red_lower, red_upper, color="red", alpha=0.10, label="Красная зона (±3σ)")
    plt.fill_between(times, yellow_lower, yellow_upper, color="yellow", alpha=0.10)
    # plt.fill_between(times, green_lower, green_upper, color="green", alpha=0.10, label="Зелёная зона (±1σ)")

    plt.plot(times, buffer_remaining, marker=".", color="tab:blue", label="Оставшийся буфер")

    # plt.fill_between(times, 0.7, 1, color="green", alpha=0.1, label="Зелёная зона")
    # plt.fill_between(times, 0.3, 0.7, color="yellow", alpha=0.1, label="Жёлтая зона")
    # plt.fill_between(times, 0, 0.3, color="red", alpha=0.1, label="Красная зона")

    for i, t in enumerate(crit_tasks):
        # plt.text(times[i*2+1], buffer_remaining[i*2+1], f"{t.task_id}", rotation=0, ha='right', fontsize=8)
        plt.text(times[i], buffer_remaining[i], f"{t.task_id}", rotation=0, ha='right', fontsize=8)

    plt.title("Сгорание буфера проекта (по критическому пути)")
    plt.xlabel("Плановое время окончания задачи")
    plt.ylabel("Оставшийся буфер")
    plt.ylim(min(0,buffer_remaining[-1]), buffer_remaining[0]+buffer_remaining[0]*0.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig("output/plots/buffer_penetration_timeline.png", dpi=300, bbox_inches="tight")






if __name__ == "__main__":
    PERCENTILE_TASK = 0.2
    PERCENTILE_PROJECT = 0.6
    PERCENTILES_RANGE = np.arange(0.05, 0.96, 0.05)
    # PERCENTILES_RANGE = [0.1, 0.5, 0.9]
    PERCENTILES_FOR_PLOT = [0.1, 0.5, 0.9]

    print("______________________________________________________")
    print(f"Started at {datetime.now().time()}")
    print("______________________________________________________")
    
    # # Нахождение буфера проекта
    pr_buffer = part1_3_project_buffer(percentile_tasks=PERCENTILE_TASK, percentile_project=PERCENTILE_PROJECT * 100)
    # # print(pr_buffer)
    # # # pr_buffer = 0
    # # # Расчет задач, построение диаграммы Гантта, экспорт таблицы задач
    scheduled_tasks, _, _ = part1_1_schedule_project(pr_buffer, percentile=PERCENTILE_TASK)
    # # Построение графиков кумулятивных функций распределения и плотности вероятности
    # part1_2_explore_percentile_effect(percentiles=PERCENTILES_RANGE)
    # # Построение Парето графика (Простои-Длительность для разных процентилей задач)
    # part1_4_plot_pareto_idle_vs_duration(PERCENTILES_RANGE)
    # # Построение графика плотности вероятности с разными процентилями
    # part1_5_multiple_percentiles(PERCENTILES_FOR_PLOT)
    # # Тепловые карты по длительности, 
    # part1_6_plot_heatmaps(task_percentiles=PERCENTILES_RANGE, project_percentiles=PERCENTILES_RANGE)

    buffer_penetration_timeline(scheduled_tasks, pr_buffer)

