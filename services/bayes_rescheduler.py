from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm

from services.scheduler import topo_sort_tasks


@dataclass
class GlobalBiasPosterior:
    mean: float   # posterior mean for theta = ln(bias)
    var: float    # posterior variance


def lognorm_params_from_mean_std(mean: float, stddev: float):
    """
    Если T ~ LogNormal(mu_log, sigma_log^2),
    то по mean/stddev в обычной шкале получаем mu_log/sigma_log.
    """
    a = 1.0 + (stddev / mean) ** 2
    sigma_log = float(np.sqrt(np.log(a)))
    mu_log = float(np.log(mean) - 0.5 * np.log(a))
    return mu_log, sigma_log


def percentile_duration(mu_log, sigma_log, percentile):
    return float(lognorm.ppf(percentile, s=sigma_log, scale=np.exp(mu_log)))


def conditional_remaining_duration(mu_log, sigma_log, spent, percentile):
    """
    Остаток длительности для задачи, которая уже выполняется spent времени:
    R = Q_p(T | T > spent) - spent
    """
    dist = lognorm(s=sigma_log, scale=np.exp(mu_log))
    fx = float(dist.cdf(spent))

    if fx >= 0.999999:
        return 0.0

    u = fx + percentile * (1.0 - fx)
    total_q = float(dist.ppf(u))
    return max(0.0, total_q - spent)


def estimate_global_bias_posterior(tasks, progress, prior_mean=0.0, prior_std=0.30, obs_noise=0.10):
    """
    Байесовская оценка общего систематического смещения:

        ln(T_fact_i) = mu_log_i + theta + eps_i
        theta ~ N(prior_mean, prior_std^2)
        eps_i ~ N(0, sigma_log_i^2 + obs_noise^2)

    progress:
    {
        task_id: {
            "status": "done" | "in_progress" | "not_started",
            "actual_duration": ...   # только для done
            "spent": ...             # только для in_progress
        }
    }
    """
    prior_var = prior_std ** 2

    precision = 1.0 / prior_var
    weighted_sum = prior_mean / prior_var

    used_tasks = 0

    for task in tasks:
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] != "done":
            continue

        actual_duration = float(info["actual_duration"])
        if actual_duration <= 0:
            continue

        mu_log, sigma_log = lognorm_params_from_mean_std(task.mean, task.stddev)
        z_i = np.log(actual_duration) - mu_log
        obs_var = sigma_log ** 2 + obs_noise ** 2

        precision += 1.0 / obs_var
        weighted_sum += z_i / obs_var
        used_tasks += 1

    post_var = 1.0 / precision
    post_mean = post_var * weighted_sum

    return GlobalBiasPosterior(mean=post_mean, var=post_var), used_tasks


def predictive_task_distribution(task, posterior):
    """
    Новое распределение будущей задачи с учетом глобального смещения:
        ln(T_new) ~ N(mu_i + posterior.mean, sigma_i^2 + posterior.var)
    """
    base_mu_log, base_sigma_log = lognorm_params_from_mean_std(task.mean, task.stddev)
    mu_pred = base_mu_log + posterior.mean
    sigma_pred = float(np.sqrt(base_sigma_log ** 2 + posterior.var))
    return mu_pred, sigma_pred


def reconstruct_fact_schedule(tasks, progress):
    """
    Восстанавливает фактический график завершённых задач от времени 0,
    если порядок выполнения соответствует исходному графу зависимостей и ресурсным ограничениям.

    Для задач:
      - done: используем actual_duration
      - in_progress / not_started: факт не достраиваем
    """
    tasks = deepcopy(tasks)
    task_map = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    role_ready = defaultdict(float)
    current_time = 0.0

    for task_id in order:
        task = task_map[task_id]
        info = progress.get(task.task_id, {"status": "not_started"})
        status = info["status"]

        if status != "done":
            continue

        dep_end = max(
            [task_map[dep].real_end_time for dep in task.dependencies if task_map[dep].real_end_time is not None],
            default=0.0
        )

        start_time = max(dep_end, role_ready[task.role])
        duration = float(info["actual_duration"])

        task.real_start_time = start_time
        task.real_duration = duration
        task.real_end_time = start_time + duration

        role_ready[task.role] = task.real_end_time
        current_time = max(current_time, task.real_end_time)

    return tasks, current_time


def build_rescheduled_plan(tasks, progress, current_time, percentile, posterior):
    """
    Строит новый план:
      - done: фиксируются по реконструированному факту
      - in_progress: начало реконструируется, остаток считается условным квантилем
      - not_started: строятся заново по обновлённым распределениям
    """
    fact_tasks, reconstructed_time = reconstruct_fact_schedule(tasks, progress)
    current_time = max(current_time, reconstructed_time)

    tasks = deepcopy(fact_tasks)
    task_map = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    role_ready = defaultdict(float)

    # сначала completed
    for task in tasks:
        info = progress.get(task.task_id, {"status": "not_started"})
        if info["status"] == "done":
            task.planned_start_time = task.real_start_time
            task.planned_duration = task.real_duration
            task.planned_end_time = task.real_end_time
            role_ready[task.role] = max(role_ready[task.role], task.planned_end_time)

    # потом in_progress
    for task_id in order:
        task = task_map[task_id]
        info = progress.get(task.task_id, {"status": "not_started"})

        if info["status"] != "in_progress":
            continue

        dep_end = max(
            [task_map[dep].planned_end_time for dep in task.dependencies],
            default=0.0
        )

        start_time = max(dep_end, role_ready[task.role])
        spent = float(info["spent"])

        mu_log, sigma_log = predictive_task_distribution(task, posterior)
        remaining = conditional_remaining_duration(mu_log, sigma_log, spent, percentile)

        task.real_start_time = start_time
        task.real_duration = spent
        task.real_end_time = start_time + spent

        task.planned_start_time = start_time
        task.planned_duration = spent + remaining
        task.planned_end_time = start_time + spent + remaining

        role_ready[task.role] = task.planned_end_time
        current_time = max(current_time, task.real_end_time)

    # затем оставшиеся not_started
    for task_id in order:
        task = task_map[task_id]
        info = progress.get(task.task_id, {"status": "not_started"})

        if info["status"] in ("done", "in_progress"):
            continue

        dep_end = max(
            [task_map[dep].planned_end_time for dep in task.dependencies],
            default=0.0
        )

        start_time = max(dep_end, role_ready[task.role], current_time)

        mu_log, sigma_log = predictive_task_distribution(task, posterior)
        duration = percentile_duration(mu_log, sigma_log, percentile)

        task.planned_start_time = start_time
        task.planned_duration = duration
        task.planned_end_time = start_time + duration

        role_ready[task.role] = task.planned_end_time

    project_finish = max(
        [t.planned_end_time for t in tasks if t.planned_end_time is not None],
        default=0.0
    )

    return tasks, project_finish, current_time


def reschedule_with_fixed_project_deadline(
    tasks,
    progress,
    target_finish_time,
    current_time=None,
    base_percentile=0.80,
    prior_mean=0.0,
    prior_std=0.30,
    obs_noise=0.10,
    min_percentile=0.05,
    max_percentile=0.99,
    max_iter=50,
    tolerance=1e-4
):
    """
    Главная функция перепланирования.

    Новая логика:
    1. По completed задачам оценивается глобальное смещение.
    2. Реконструируется факт завершённых задач от 0.
    3. Для оставшихся задач сдвигаются распределения.
    4. Подбирается МАКСИМАЛЬНЫЙ percentile, при котором:
           project_finish <= target_finish_time

    То есть план старается быть как можно ближе к дедлайну, не нарушая его.
    """
    posterior, used_tasks = estimate_global_bias_posterior(
        tasks=tasks,
        progress=progress,
        prior_mean=prior_mean,
        prior_std=prior_std,
        obs_noise=obs_noise
    )

    reconstructed_tasks, reconstructed_time = reconstruct_fact_schedule(tasks, progress)
    if current_time is None:
        current_time = reconstructed_time

    # 1. Проверка на достижимость дедлайна при минимальном percentile
    min_tasks, min_finish, current_time = build_rescheduled_plan(
        tasks=tasks,
        progress=progress,
        current_time=current_time,
        percentile=min_percentile,
        posterior=posterior
    )

    if min_finish > target_finish_time:
        return {
            "tasks": min_tasks,
            "project_finish": min_finish,
            "current_time": current_time,
            "used_percentile": min_percentile,
            "deadline_met": False,
            "posterior": posterior,
            "used_observations": used_tasks,
            "bias_factor_mean": float(np.exp(posterior.mean)),
            "message": "Даже при минимальном процентиле сохранить дедлайн невозможно."
        }

    # 2. Проверка верхней границы:
    # если даже при максимально большом percentile проект все еще не выходит за дедлайн,
    # то это и есть лучшее решение: взять max_percentile
    max_tasks, max_finish, _ = build_rescheduled_plan(
        tasks=tasks,
        progress=progress,
        current_time=current_time,
        percentile=max_percentile,
        posterior=posterior
    )

    if max_finish <= target_finish_time:
        return {
            "tasks": max_tasks,
            "project_finish": max_finish,
            "current_time": current_time,
            "used_percentile": max_percentile,
            "deadline_met": True,
            "posterior": posterior,
            "used_observations": used_tasks,
            "bias_factor_mean": float(np.exp(posterior.mean)),
            "message": "Проект укладывается в дедлайн даже при максимальном процентиле; выбран максимально близкий к дедлайну план."
        }

    # 3. Иначе бинарным поиском ищем максимальный допустимый percentile
    lo, hi = min_percentile, max_percentile

    best_tasks = min_tasks
    best_finish = min_finish
    best_p = min_percentile

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        trial_tasks, trial_finish, _ = build_rescheduled_plan(
            tasks=tasks,
            progress=progress,
            current_time=current_time,
            percentile=mid,
            posterior=posterior
        )

        if trial_finish <= target_finish_time:
            # percentile допустим, пробуем поднять еще выше
            best_tasks = trial_tasks
            best_finish = trial_finish
            best_p = mid
            lo = mid
        else:
            # percentile слишком большой, план вылез за дедлайн
            hi = mid

        if hi - lo < tolerance:
            break

    return {
        "tasks": best_tasks,
        "project_finish": best_finish,
        "current_time": current_time,
        "used_percentile": best_p,
        "deadline_met": True,
        "posterior": posterior,
        "used_observations": used_tasks,
        "bias_factor_mean": float(np.exp(posterior.mean)),
        "message": "Перепланирование выполнено; выбран максимальный допустимый percentile, дающий срок проекта максимально близко к дедлайну."
    }