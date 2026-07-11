"""
Сравнение байесовского перепланировщика (жадный порядок) vs OR-Tools CP-SAT
при различных сценариях отклонений расписания.

Оба подхода используют ОДИНАКОВЫЙ байесовский постериор для оценки системного
смещения длительностей. Различие одно:

  Bayesian  — оставшиеся задачи упорядочиваются жадно (BFS-топологически),
               затем бинарным поиском находится максимальный персентиль,
               при котором дедлайн ещё достижим.

  OR-Tools  — оставшиеся задачи оптимально упорядочиваются CP-SAT-решателем,
               минимизируя makespan при тех же байес-скорректированных
               длительностях. Бинарный поиск персентиля — идентичный.

Тестируемые сценарии
--------------------
  baseline, global_overrun_1.5x / _2x, global_underrun, high_variance,
  role_bottleneck, cascading_delay, early_wins_late_slip, deadline_impossible

Точки наблюдения  : 25 % / 50 % / 75 % выполненных задач
Семена            : 5 на каждую (scenario, obs_fraction) пару

Выходные данные
---------------
  output/bayes_vs_ortools/
    results.csv          — сырые данные по каждому запуску
    heatmap_rate.png     — процент «дедлайн выполнен» OR-Tools - Байес
    heatmap_finish.png   — разница средних project_finish (дни)
    boxplots.png         — распределение finish_vs_deadline по сценариям
    bar_improvement.png  — среднее улучшение OR-Tools над Байес (дни)
    percentile_comp.png  — использованный персентиль Байес vs OR-Tools
    summary.xlsx         — сводная таблица

Запуск
------
    python -m experiments.bayes_vs_ortools
"""

from __future__ import annotations

import io
import sys

# UTF-8 вывод на Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from scipy.stats import lognorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.scenario_generator import SCENARIOS, load_scheduled_tasks
from models.task import Task
from services.bayes_rescheduler import (
    GlobalBiasPosterior,
    estimate_global_bias_posterior,
    reconstruct_fact_schedule,
    build_rescheduled_plan,
    reschedule_with_fixed_project_deadline,
    _predictive_params,
)
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule, topo_sort_tasks
from utils.lognormal import lognorm_log_params


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

TASK_FILE   = "data/tasks.csv"
OUTPUT_DIR  = Path("output/bayes_vs_ortools")

SCENARIOS_SUBSET = [
    "baseline",
    "global_overrun_1.5x",
    "global_overrun_2x",
    "global_underrun",
    "high_variance",
    "role_bottleneck",
    "cascading_delay",
    "early_wins_late_slip",
    "deadline_impossible",
]

BASE_PERCENTILE      = 0.9
PLAN_SEED            = 42
OBSERVATION_FRACTIONS = [0.25, 0.50, 0.75]
N_SEEDS              = 5

SCALE         = 100    # целочисленное масштабирование для CP-SAT
CPSAT_TIMEOUT = 1.0    # секунд на один вызов CP-SAT в бинарном поиске
MAX_ITER      = 20     # итераций бинарного поиска


# ---------------------------------------------------------------------------
# OR-Tools: оптимальный план оставшихся задач
# ---------------------------------------------------------------------------

def _ortools_plan_remaining(
    fact_tasks: list[Task],
    progress: dict[int, dict],
    current_time: float,
    percentile: float,
    posterior: GlobalBiasPosterior,
    scale: int = SCALE,
    timeout: float = CPSAT_TIMEOUT,
) -> tuple[list[Task], float]:
    """
    Оптимально расписывает невыполненные задачи через CP-SAT.

    Завершённые задачи берутся как зафиксированные (из factual schedule).
    Для каждой оставшейся задачи вычисляется байес-скорректированная
    длительность при данном персентиле; CP-SAT минимизирует makespan.

    Returns:
        (result_tasks, project_finish_float)
        result_tasks: список Task с проставленными planned_* полями.
        project_finish_float: максимальный planned_end_time.
    """
    task_map = {t.task_id: t for t in fact_tasks}

    done_ids = {
        tid for tid, info in progress.items()
        if info.get("status") == "done"
    }
    not_started = [t for t in fact_tasks if t.task_id not in done_ids]

    # --- Если все задачи выполнены ---
    if not not_started:
        finish = max(
            (t.real_end_time for t in fact_tasks
             if t.real_end_time is not None),
            default=current_time,
        )
        return deepcopy(fact_tasks), finish

    # --- Байес-скорректированные длительности ---
    durations: dict[int, float] = {}
    for t in not_started:
        mu_pred, sigma_pred = _predictive_params(t, posterior)
        dur = float(lognorm.ppf(percentile, s=sigma_pred, scale=np.exp(mu_pred)))
        durations[t.task_id] = max(dur, 0.01)

    # --- Доступность ролей из выполненных задач ---
    role_ready_float: dict[str, float] = defaultdict(float)
    done_end_float: dict[int, float] = {}
    for tid in done_ids:
        t = task_map[tid]
        if t.real_end_time is not None:
            role_ready_float[t.role] = max(role_ready_float[t.role], t.real_end_time)
            done_end_float[tid] = t.real_end_time

    # --- Горизонт ---
    total_rem = sum(round(d * scale) for d in durations.values())
    cur_int   = round(current_time * scale)
    horizon   = cur_int + total_rem + round(max(role_ready_float.values(), default=0) * scale) + 1

    # --- CP-SAT модель ---
    model = cp_model.CpModel()
    starts_var:    dict[int, cp_model.IntVar] = {}
    ends_var:      dict[int, cp_model.IntVar] = {}
    intervals_var: dict[int, cp_model.IntervalVar] = {}

    for t in not_started:
        dur_int = max(1, round(durations[t.task_id] * scale))

        # Нижняя граница: current_time, если задача блокируется выполненной
        has_done_pred = any(dep in done_ids for dep in t.dependencies)
        effective_floor = current_time if has_done_pred else 0.0

        # Роль освободится не раньше последней выполненной задачи той же роли
        role_fl = role_ready_float.get(t.role, 0.0)

        lb = max(round(effective_floor * scale), round(role_fl * scale))

        s  = model.new_int_var(lb, horizon, f"s_{t.task_id}")
        e  = model.new_int_var(lb + dur_int, horizon + dur_int, f"e_{t.task_id}")
        iv = model.new_interval_var(s, dur_int, e, f"iv_{t.task_id}")
        starts_var[t.task_id]    = s
        ends_var[t.task_id]      = e
        intervals_var[t.task_id] = iv

    # --- Ограничения предшествования ---
    for t in not_started:
        for dep_id in t.dependencies:
            if dep_id in done_end_float:
                # Зависимость от выполненной — жёсткий нижний предел
                model.add(starts_var[t.task_id] >= round(done_end_float[dep_id] * scale))
            elif dep_id in starts_var:
                # Зависимость от ещё не начатой
                model.add(starts_var[t.task_id] >= ends_var[dep_id])
            # (задача из progress, но не "done" — нет в словарях; пропускаем)

    # --- Ресурсные ограничения (без перекрытий) ---
    role_ivals: dict[str, list] = defaultdict(list)
    for t in not_started:
        role_ivals[t.role].append(intervals_var[t.task_id])
    for ivals in role_ivals.values():
        if len(ivals) > 1:
            model.add_no_overlap(ivals)

    # --- Минимизация makespan ---
    makespan_var = model.new_int_var(0, horizon + 10000, "makespan")
    model.add_max_equality(makespan_var, list(ends_var.values()))
    model.minimize(makespan_var)

    # --- Решение ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers  = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], float("inf")

    opt_makespan = solver.value(makespan_var) / scale

    # --- Строим результирующий список задач ---
    result_tasks = deepcopy(fact_tasks)
    result_map   = {t.task_id: t for t in result_tasks}

    # Зафиксировать выполненные
    for tid in done_ids:
        t = result_map[tid]
        t.planned_start_time = t.real_start_time
        t.planned_duration   = t.real_duration
        t.planned_end_time   = t.real_end_time

    # Заполнить оставшиеся из CP-SAT
    for t in not_started:
        rt    = result_map[t.task_id]
        start = solver.value(starts_var[t.task_id]) / scale
        dur   = durations[t.task_id]
        rt.planned_start_time = start
        rt.planned_duration   = dur
        rt.planned_end_time   = start + dur

    return result_tasks, opt_makespan


# ---------------------------------------------------------------------------
# OR-Tools: бинарный поиск персентиля (аналог reschedule_with_fixed_project_deadline)
# ---------------------------------------------------------------------------

def reschedule_ortools(
    tasks: list[Task],
    progress: dict[int, dict],
    target_finish_time: float,
    current_time: float | None = None,
    base_percentile: float = BASE_PERCENTILE,
    prior_mean: float = 0.0,
    prior_std: float = 0.30,
    obs_noise: float = 0.10,
    min_percentile: float = 0.05,
    max_percentile: float = 0.99,
    max_iter: int = MAX_ITER,
    tolerance: float = 1e-3,
    scale: int = SCALE,
    timeout: float = CPSAT_TIMEOUT,
) -> dict:
    """
    Байесовская оценка смещения + OR-Tools CP-SAT для расписания оставшихся задач.

    Алгоритм идентичен Bayesian (тот же постериор, тот же бинарный поиск),
    кроме одного: вместо жадного планировщика вызывается CP-SAT.
    """
    # 1. Постериор (тот же, что у Байес-подхода)
    posterior, used = estimate_global_bias_posterior(
        tasks, progress, prior_mean, prior_std, obs_noise
    )

    # 2. Факт. расписание (для установки current_time)
    fact_tasks, reconstructed_time = reconstruct_fact_schedule(tasks, progress)
    if current_time is None:
        current_time = reconstructed_time

    def _plan(p: float) -> float:
        _, finish = _ortools_plan_remaining(
            fact_tasks, progress, current_time, p, posterior, scale, timeout
        )
        return finish

    def _result(p, finish, met, msg):
        return {
            "project_finish":   finish,
            "current_time":     current_time,
            "used_percentile":  p,
            "deadline_met":     met,
            "posterior":        posterior,
            "used_observations": used,
            "bias_factor_mean": float(np.exp(posterior.mean)),
            "message":          msg,
        }

    # 3. Проверяем base_percentile
    base_finish = _plan(base_percentile)

    # Все задачи выполнены
    if abs(base_finish - current_time) < 1e-9:
        return _result(
            base_percentile, base_finish,
            base_finish <= target_finish_time,
            "All tasks complete.",
        )

    if base_finish <= target_finish_time:
        # Дедлайн соблюдён; если смещение отрицательное — ищем выше
        if posterior.mean < -0.05:
            hi_finish = _plan(max_percentile)
            if hi_finish <= target_finish_time:
                return _result(max_percentile, hi_finish, True,
                               "Negative bias; raised to max.")
            lo, hi = base_percentile, max_percentile
            best_p, best_f = base_percentile, base_finish
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                f = _plan(mid)
                if f <= target_finish_time:
                    best_p, best_f = mid, f
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tolerance:
                    break
            return _result(best_p, best_f, True, "Negative bias upward search.")
        return _result(base_percentile, base_finish, True, "No significant bias.")

    # 4. Дедлайн нарушен при base — ищем вниз
    min_finish = _plan(min_percentile)
    if min_finish > target_finish_time:
        return _result(min_percentile, min_finish, False, "Deadline unachievable.")

    lo, hi = min_percentile, base_percentile
    best_p, best_f = min_percentile, min_finish
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = _plan(mid)
        if f <= target_finish_time:
            best_p, best_f = mid, f
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance:
            break

    return _result(best_p, best_f, True,
                   f"Positive bias; lowered percentile to {best_p:.3f}.")


# ---------------------------------------------------------------------------
# Один запуск (scenario, n_observed, seed)
# ---------------------------------------------------------------------------

def run_single(
    scenario_name: str,
    n_observed: int,
    seed: int,
    scheduled_tasks: list[Task],
    target_finish_time: float,
    task_file: str = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
) -> dict:
    """
    Запускает оба перепланировщика для одной (scenario, obs, seed) комбинации.

    Returns:
        dict с метриками для Байес и OR-Tools.
    """
    scenario = SCENARIOS[scenario_name]
    n_tasks  = len(scheduled_tasks)

    # Генерируем прогресс
    progress = scenario["fn"](
        scheduled_tasks,
        n_observed=n_observed,
        seed=seed,
        **scenario["kwargs"],
    )

    # --- Байесовский перепланировщик ---
    fresh_bayes = load_tasks_from_csv(task_file)
    build_schedule(fresh_bayes, percentile=base_percentile, seed=PLAN_SEED)

    t0 = time.perf_counter()
    try:
        res_b = reschedule_with_fixed_project_deadline(
            tasks=fresh_bayes,
            progress=progress,
            target_finish_time=target_finish_time,
            base_percentile=base_percentile,
        )
        bayes_ok = True
    except Exception as exc:
        res_b = {}
        bayes_ok = False
        bayes_err = str(exc)
    t_bayes = time.perf_counter() - t0

    # --- OR-Tools перепланировщик ---
    fresh_ort = load_tasks_from_csv(task_file)
    build_schedule(fresh_ort, percentile=base_percentile, seed=PLAN_SEED)

    t0 = time.perf_counter()
    try:
        res_o = reschedule_ortools(
            tasks=fresh_ort,
            progress=progress,
            target_finish_time=target_finish_time,
            base_percentile=base_percentile,
        )
        ort_ok = True
    except Exception as exc:
        res_o = {}
        ort_ok = False
        ort_err = str(exc)
    t_ort = time.perf_counter() - t0

    base = {
        "scenario":    scenario_name,
        "description": scenario["description"],
        "n_observed":  n_observed,
        "obs_fraction": round(n_observed / n_tasks, 2),
        "seed":        seed,
    }

    if not bayes_ok:
        b_row = dict(bayes_finish=np.nan, bayes_met=False,
                     bayes_pct=np.nan, bayes_bias=np.nan,
                     bayes_time=t_bayes, bayes_error=bayes_err)
    else:
        b_row = dict(
            bayes_finish=round(res_b["project_finish"], 3),
            bayes_met=res_b["deadline_met"],
            bayes_pct=round(res_b["used_percentile"], 4),
            bayes_bias=round(res_b["bias_factor_mean"], 4),
            bayes_time=round(t_bayes, 3),
            bayes_error=None,
        )

    if not ort_ok:
        o_row = dict(ort_finish=np.nan, ort_met=False,
                     ort_pct=np.nan, ort_bias=np.nan,
                     ort_time=t_ort, ort_error=ort_err)
    else:
        o_row = dict(
            ort_finish=round(res_o["project_finish"], 3),
            ort_met=res_o["deadline_met"],
            ort_pct=round(res_o["used_percentile"], 4),
            ort_bias=round(res_o["bias_factor_mean"], 4),
            ort_time=round(t_ort, 3),
            ort_error=None,
        )

    row = {**base, **b_row, **o_row}

    # Дельта (>0 означает, что OR-Tools даёт более ранний финиш)
    if bayes_ok and ort_ok:
        row["improvement_days"] = round(
            res_b["project_finish"] - res_o["project_finish"], 3
        )
        row["pct_improvement"] = round(
            res_o["used_percentile"] - res_b["used_percentile"], 4
        )
        row["both_met"]      = res_b["deadline_met"] and res_o["deadline_met"]
        row["only_ort_met"]  = (not res_b["deadline_met"]) and res_o["deadline_met"]
        row["only_bayes_met"] = res_b["deadline_met"] and (not res_o["deadline_met"])
    else:
        row["improvement_days"] = np.nan
        row["pct_improvement"]  = np.nan
        row["both_met"]         = False
        row["only_ort_met"]     = False
        row["only_bayes_met"]   = False

    return row


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def run_all(
    task_file:             str   = TASK_FILE,
    base_percentile:       float = BASE_PERCENTILE,
    plan_seed:             int   = PLAN_SEED,
    observation_fractions: list  = OBSERVATION_FRACTIONS,
    n_seeds:               int   = N_SEEDS,
    scenarios:             list  = SCENARIOS_SUBSET,
    output_dir:            Path  = OUTPUT_DIR,
    verbose:               bool  = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduled_tasks = load_scheduled_tasks(task_file, base_percentile, plan_seed)
    n_tasks         = len(scheduled_tasks)
    target_finish   = max(t.planned_end_time for t in scheduled_tasks)

    obs_counts = [max(1, round(f * n_tasks)) for f in observation_fractions]
    total_runs = len(scenarios) * len(obs_counts) * n_seeds

    if verbose:
        print(f"Задач: {n_tasks} | Дедлайн: {target_finish:.1f} дн. | "
              f"Сценарии: {len(scenarios)} | "
              f"Запусков: {total_runs}")
        print()

    rows = []
    done = 0
    t_start = time.perf_counter()

    for sc in scenarios:
        if sc not in SCENARIOS:
            print(f"  [SKIP] {sc}")
            continue
        for n_obs in obs_counts:
            for seed in range(n_seeds):
                row = run_single(
                    scenario_name=sc,
                    n_observed=n_obs,
                    seed=seed,
                    scheduled_tasks=scheduled_tasks,
                    target_finish_time=target_finish,
                    task_file=task_file,
                    base_percentile=base_percentile,
                )
                rows.append(row)
                done += 1
                if verbose:
                    imp = row.get("improvement_days", float("nan"))
                    imp_str = f"{imp:+.1f} дн." if not np.isnan(imp) else "ERR"
                    b_met = "Y" if row.get("bayes_met") else "N"
                    o_met = "Y" if row.get("ort_met")   else "N"
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / done * (total_runs - done) if done > 0 else 0
                    print(f"  [{done:3d}/{total_runs}] {sc:<25} "
                          f"obs={n_obs:2d} s={seed} "
                          f"Bayes={b_met} ORT={o_met} "
                          f"Δ={imp_str:>10}  "
                          f"ETA {eta/60:.1f}m")

    df = pd.DataFrame(rows)

    # Сохранение сырых данных
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nCSV -> {csv_path}")

    return df, target_finish, n_tasks


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

SCENARIO_SHORT = {
    "baseline":              "baseline",
    "global_overrun_1.5x":  "overrun_1.5x",
    "global_overrun_2x":    "overrun_2x",
    "global_underrun":      "underrun",
    "high_variance":        "hi_variance",
    "role_bottleneck":      "bottleneck",
    "cascading_delay":      "cascading",
    "early_wins_late_slip": "early_wins",
    "deadline_impossible":  "impossible",
}

def _frac_label(frac: float) -> str:
    """Approximate fraction -> label (handles rounding from n_tasks)."""
    for ref, label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
        if abs(frac - ref) < 0.05:
            return label
    return f"{frac:.0%}"

FRAC_LABELS: dict = {}   # populated lazily from data


def _clean(df: pd.DataFrame, frac: float | None = None, tol: float = 0.06) -> pd.DataFrame:
    """Return error-free rows, optionally filtered by approximate obs_fraction."""
    mask = (df["bayes_error"].isna()) & (df["ort_error"].isna())
    if frac is not None:
        mask &= (df["obs_fraction"] - frac).abs() < tol
    return df[mask].copy()


def _nearest_frac(df: pd.DataFrame, target: float) -> float:
    """Return the actual obs_fraction value closest to target."""
    fracs = df["obs_fraction"].unique()
    return float(min(fracs, key=lambda f: abs(f - target)))


def plot_heatmaps(df: pd.DataFrame, target_finish: float, output_dir: Path) -> None:
    """
    2×3 тепловые карты:
      - Верхний ряд: OR-Tools deadline_met_rate - Bayes deadline_met_rate
      - Нижний ряд: Bayes_finish - OR-Tools_finish (среднее, дни)
    Столбцы: наблюдение 25 % / 50 % / 75 %
    """
    fracs   = sorted(df["obs_fraction"].unique())
    scens   = [s for s in SCENARIOS_SUBSET if s in df["scenario"].unique()]
    n_sc    = len(scens)
    n_fr    = len(fracs)

    fig, axes = plt.subplots(2, n_fr, figsize=(5 * n_fr, max(5, n_sc * 0.55 + 1.5)))
    fig.suptitle(
        "OR-Tools CP-SAT vs байесовский перепланировщик\n"
        "Верхний ряд: Δ доля выполнения дедлайна (OR-Tools - Bayes)\n"
        "Нижний ряд: Δ project_finish (Bayes - OR-Tools, дни; > 0 = OR-Tools лучше)",
        fontsize=10, fontweight="bold",
    )

    # Строим матрицы
    rate_mat = np.full((n_sc, n_fr), np.nan)
    days_mat = np.full((n_sc, n_fr), np.nan)

    for fi, frac in enumerate(fracs):
        sub = _clean(df, frac)
        for si, sc in enumerate(scens):
            sg = sub[sub["scenario"] == sc]
            if sg.empty:
                continue
            rate_mat[si, fi] = (
                sg["ort_met"].mean() - sg["bayes_met"].mean()
            )
            days_mat[si, fi] = sg["improvement_days"].mean()

    ylabels = [SCENARIO_SHORT.get(s, s) for s in scens]

    for fi, frac in enumerate(fracs):
        # Верхний ряд: rate diff
        ax = axes[0, fi]
        im = ax.imshow(rate_mat[:, fi:fi+1], aspect="auto",
                       cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([f"obs={_frac_label(frac)}"])
        ax.set_yticks(range(n_sc))
        ax.set_yticklabels(ylabels if fi == 0 else [], fontsize=8)
        ax.set_title(f"obs={_frac_label(frac)}", fontsize=9)
        for si in range(n_sc):
            val = rate_mat[si, fi]
            if not np.isnan(val):
                col = "black" if abs(val) < 0.35 else "white"
                ax.text(0, si, f"{val:+.2f}", ha="center", va="center",
                        fontsize=8, color=col, fontweight="bold")
        if fi == n_fr - 1:
            plt.colorbar(im, ax=ax, label="delta met rate")

        # Нижний ряд: days diff
        ax = axes[1, fi]
        lim = np.nanmax(np.abs(days_mat)) or 1
        im2 = ax.imshow(days_mat[:, fi:fi+1], aspect="auto",
                        cmap="RdYlGn", vmin=-lim, vmax=lim)
        ax.set_xticks([0])
        ax.set_xticklabels([f"obs={_frac_label(frac)}"])
        ax.set_yticks(range(n_sc))
        ax.set_yticklabels(ylabels if fi == 0 else [], fontsize=8)
        for si in range(n_sc):
            val = days_mat[si, fi]
            if not np.isnan(val):
                col = "black" if abs(val) < lim * 0.6 else "white"
                ax.text(0, si, f"{val:+.1f}", ha="center", va="center",
                        fontsize=8, color=col, fontweight="bold")
        if fi == n_fr - 1:
            plt.colorbar(im2, ax=ax, label="Δ finish (days)")

    plt.tight_layout()
    path = output_dir / "heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  heatmaps -> {path}")


def plot_improvement_bars(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Барные графики: среднее улучшение (дни) OR-Tools над Байес.
    Три панели = три точки наблюдения.
    """
    fracs = sorted(df["obs_fraction"].unique())
    scens = [s for s in SCENARIOS_SUBSET if s in df["scenario"].unique()]

    fig, axes = plt.subplots(1, len(fracs),
                             figsize=(6 * len(fracs), max(4, len(scens) * 0.45 + 1.5)),
                             sharey=True)
    fig.suptitle(
        "Улучшение OR-Tools над байесовским подходом\n"
        "(> 0 = OR-Tools даёт более ранний финиш проекта)",
        fontsize=11, fontweight="bold",
    )

    y = np.arange(len(scens))
    ylabels = [SCENARIO_SHORT.get(s, s) for s in scens]

    for fi, (ax, frac) in enumerate(zip(axes, fracs)):
        sub = _clean(df, frac)
        vals = []
        errs = []
        for sc in scens:
            sg = sub[sub["scenario"] == sc]["improvement_days"].dropna()
            vals.append(sg.mean() if len(sg) else 0.0)
            errs.append(sg.std()  if len(sg) > 1 else 0.0)

        colors = ["#5DBB63" if v >= 0.05 else
                  "#E84040" if v <= -0.05 else "#AAAAAA"
                  for v in vals]

        ax.barh(y, vals, xerr=errs, color=colors, alpha=0.85,
                edgecolor="white", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels if fi == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Δ финиш (дни)", fontsize=9)
        ax.set_title(f"Наблюдение {_frac_label(frac)}", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for yi, (v, e) in enumerate(zip(vals, errs)):
            if abs(v) > 0.01:
                ax.text(v + (0.05 if v >= 0 else -0.05), yi,
                        f"{v:+.1f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7.5)

    handles = [
        mpatches.Patch(color="#5DBB63", label="OR-Tools лучше"),
        mpatches.Patch(color="#E84040", label="Bayes лучше"),
        mpatches.Patch(color="#AAAAAA", label="Равны"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    path = output_dir / "bar_improvement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  bar_improvement -> {path}")


def plot_deadline_rate_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Сгруппированный барный график: доля выполнения дедлайна Bayes vs OR-Tools.
    """
    fracs = sorted(df["obs_fraction"].unique())
    scens = [s for s in SCENARIOS_SUBSET if s in df["scenario"].unique()]

    fig, axes = plt.subplots(1, len(fracs),
                             figsize=(6 * len(fracs), max(4, len(scens) * 0.5 + 1.5)),
                             sharey=True)
    fig.suptitle(
        "Доля выполнения дедлайна: Bayes (синий) vs OR-Tools (оранжевый)",
        fontsize=11, fontweight="bold",
    )

    bwidth = 0.38
    y      = np.arange(len(scens))
    ylabels = [SCENARIO_SHORT.get(s, s) for s in scens]

    for fi, (ax, frac) in enumerate(zip(axes, fracs)):
        sub = _clean(df, frac)
        b_rates, o_rates = [], []
        for sc in scens:
            sg = sub[sub["scenario"] == sc]
            b_rates.append(sg["bayes_met"].mean() if len(sg) else 0)
            o_rates.append(sg["ort_met"].mean()   if len(sg) else 0)

        ax.barh(y - bwidth/2, b_rates, bwidth, color="#4C9BE8", alpha=0.85,
                label="Bayes", edgecolor="white")
        ax.barh(y + bwidth/2, o_rates, bwidth, color="#F28C38", alpha=0.85,
                label="OR-Tools", edgecolor="white")

        ax.axvline(0.8, color="green", linestyle="--", linewidth=1,
                   alpha=0.7, label="80% цель")
        ax.set_xlim(0, 1.12)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels if fi == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Доля выполнения дедлайна", fontsize=9)
        ax.set_title(f"Наблюдение {_frac_label(frac)}", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        for yi, (b, o) in enumerate(zip(b_rates, o_rates)):
            ax.text(b + 0.01, yi - bwidth/2, f"{b:.0%}", va="center", fontsize=6.5)
            ax.text(o + 0.01, yi + bwidth/2, f"{o:.0%}", va="center", fontsize=6.5)

        if fi == 0:
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    path = output_dir / "deadline_rate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  deadline_rate -> {path}")


def plot_percentile_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter plot: использованный персентиль Bayes vs OR-Tools.
    Точки выше диагонали — OR-Tools использует более высокий персентиль.
    """
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())

    cmap   = plt.get_cmap("viridis")
    colors = {f: cmap(i / max(len(fracs) - 1, 1)) for i, f in enumerate(fracs)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for frac in fracs:
        sub = clean[clean["obs_fraction"] == frac]
        ax.scatter(
            sub["bayes_pct"], sub["ort_pct"],
            c=[colors[frac]] * len(sub),
            label=f"obs={_frac_label(frac)}",
            s=30, alpha=0.6, edgecolors="none",
        )

    lim = max(clean["bayes_pct"].max(), clean["ort_pct"].max()) + 0.02
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("Персентиль Bayesian", fontsize=11)
    ax.set_ylabel("Персентиль OR-Tools", fontsize=11)
    ax.set_title(
        "Персентиль: Bayesian vs OR-Tools\n"
        "(выше диагонали = OR-Tools использует более консервативный буфер)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    plt.tight_layout()
    path = output_dir / "percentile_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  percentile_scatter -> {path}")


def plot_finish_boxplots(df: pd.DataFrame, target_finish: float,
                         output_dir: Path) -> None:
    """
    Боксплоты finish_vs_deadline для каждого сценария при obs=50%.
    Синий = Bayes, оранжевый = OR-Tools.
    """
    mid_frac = _nearest_frac(df, 0.50)
    sub = _clean(df, mid_frac)
    scens = [s for s in SCENARIOS_SUBSET if s in sub["scenario"].unique()]

    fig, ax = plt.subplots(figsize=(14, max(5, len(scens) * 0.8 + 1.5)))

    positions_b = np.arange(len(scens)) * 3
    positions_o = positions_b + 1.0

    data_b = [sub[sub["scenario"] == s]["bayes_finish"].dropna().values for s in scens]
    data_o = [sub[sub["scenario"] == s]["ort_finish"].dropna().values   for s in scens]

    bp_b = ax.boxplot(data_b, positions=positions_b, widths=0.7,
                      patch_artist=True, notch=False,
                      medianprops=dict(color="navy", linewidth=2))
    bp_o = ax.boxplot(data_o, positions=positions_o, widths=0.7,
                      patch_artist=True, notch=False,
                      medianprops=dict(color="darkred", linewidth=2))

    for patch in bp_b["boxes"]:
        patch.set_facecolor("#4C9BE8")
        patch.set_alpha(0.75)
    for patch in bp_o["boxes"]:
        patch.set_facecolor("#F28C38")
        patch.set_alpha(0.75)

    ax.axhline(target_finish, color="red", linestyle="--",
               linewidth=1.5, label=f"Дедлайн ({target_finish:.0f} дн.)")

    mid_pos = positions_b + 0.5
    ax.set_xticks(mid_pos)
    ax.set_xticklabels(
        [SCENARIO_SHORT.get(s, s) for s in scens],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_ylabel("Project finish (дни)", fontsize=10)
    ax.set_title(f"Финиш проекта по сценариям (obs~{mid_frac:.0%})\nСиний = Bayesian, Оранжевый = OR-Tools",
                 fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        mpatches.Patch(color="#4C9BE8", label="Bayesian (жадный)"),
        mpatches.Patch(color="#F28C38", label="OR-Tools CP-SAT"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Дедлайн"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    plt.tight_layout()
    path = output_dir / "finish_boxplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  finish_boxplots -> {path}")


# ---------------------------------------------------------------------------
# Текстовый отчёт
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, target_finish: float) -> None:
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())

    print()
    print("=" * 75)
    print("ИТОГОВОЕ СРАВНЕНИЕ: Bayesian (жадный) vs OR-Tools CP-SAT")
    print("=" * 75)
    print(f"Базовый дедлайн: {target_finish:.1f} дней")
    print(f"Всего запусков:  {len(clean)}")
    print()

    for frac in fracs:
        sub = clean[clean["obs_fraction"] == frac]
        print(f"--- Наблюдение {_frac_label(frac)} ({len(sub)} запусков) ---")
        print(f"  Bayes   deadline_met: {sub['bayes_met'].mean():.1%}")
        print(f"  OR-Tools deadline_met: {sub['ort_met'].mean():.1%}")
        print(f"  Среднее улучшение OR-Tools: {sub['improvement_days'].mean():+.2f} дн.")
        print(f"  Медиана улучшения:          {sub['improvement_days'].median():+.2f} дн.")
        print(f"  Макс. улучшение:            {sub['improvement_days'].max():+.2f} дн.")
        print(f"  Только OR-Tools уложился:   {sub['only_ort_met'].sum()}")
        print(f"  Только Bayes уложился:      {sub['only_bayes_met'].sum()}")
        print()

    mid_frac = _nearest_frac(clean, 0.50)
    print(f"По сценариям (obs~{mid_frac:.0%}):")
    mid = _clean(clean, mid_frac)
    summary = (
        mid.groupby("scenario")
        .agg(
            bayes_rate  =("bayes_met",       "mean"),
            ort_rate    =("ort_met",         "mean"),
            delta_rate  =("only_ort_met",    "mean"),
            imp_mean    =("improvement_days","mean"),
            imp_max     =("improvement_days","max"),
        )
        .round(3)
        .sort_values("imp_mean", ascending=False)
    )
    print(summary.rename(columns={
        "bayes_rate":  "Bayes_met%",
        "ort_rate":    "ORT_met%",
        "delta_rate":  "+ORT_only%",
        "imp_mean":    "Δdays_mean",
        "imp_max":     "Δdays_max",
    }).to_string())
    print("=" * 75)


# ---------------------------------------------------------------------------
# Excel-сводка
# ---------------------------------------------------------------------------

def export_excel(df: pd.DataFrame, target_finish: float, output_dir: Path) -> None:
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())

    xlsx_path = output_dir / "summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw", index=False)

        rows = []
        for sc in SCENARIOS_SUBSET:
            for frac in fracs:
                sub = clean[(clean["scenario"] == sc) & (clean["obs_fraction"] == frac)]
                if sub.empty:
                    continue
                rows.append({
                    "scenario":     sc,
                    "obs_fraction": frac,
                    "n_runs":       len(sub),
                    "bayes_met":    round(sub["bayes_met"].mean(), 3),
                    "ort_met":      round(sub["ort_met"].mean(), 3),
                    "delta_rate":   round(sub["ort_met"].mean() - sub["bayes_met"].mean(), 3),
                    "imp_mean":     round(sub["improvement_days"].mean(), 2),
                    "imp_std":      round(sub["improvement_days"].std(), 2),
                    "imp_max":      round(sub["improvement_days"].max(), 2),
                    "pct_imp_mean": round(sub["pct_improvement"].mean(), 4),
                    "bayes_time_s": round(sub["bayes_time"].mean(), 3),
                    "ort_time_s":   round(sub["ort_time"].mean(), 3),
                })

        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

    print(f"  Excel -> {xlsx_path}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",   default=TASK_FILE)
    parser.add_argument("--out",     default=str(OUTPUT_DIR))
    parser.add_argument("--seeds",   default=N_SEEDS,     type=int)
    parser.add_argument("--timeout", default=CPSAT_TIMEOUT, type=float,
                        help="CP-SAT timeout per binary-search call (sec)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("=" * 65)
    print("Bayesian перепланировщик vs OR-Tools CP-SAT")
    print("=" * 65)

    df, target_finish, n_tasks = run_all(
        task_file=args.tasks,
        n_seeds=args.seeds,
        output_dir=out_dir,
        verbose=True,
    )

    print("\nСтроим графики...")
    plot_heatmaps(df, target_finish, out_dir)
    plot_improvement_bars(df, out_dir)
    plot_deadline_rate_comparison(df, out_dir)
    plot_percentile_comparison(df, out_dir)
    plot_finish_boxplots(df, target_finish, out_dir)
    export_excel(df, target_finish, out_dir)

    print_summary(df, target_finish)
