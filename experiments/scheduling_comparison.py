"""
Сравнение трёх стратегий динамического перепланирования.

════════════════════════════════════════════════════════════════
УПРАВЛЕНЧЕСКАЯ ХАРАКТЕРИСТИКА, КОТОРУЮ УЛУЧШАЕТ БАЙЕС-МОДЕЛЬ
════════════════════════════════════════════════════════════════

Байесовский перепланировщик целенаправленно улучшает
**Calibrated Schedule Buffer Reliability** —
надёжность адаптивного буфера расписания:
способность поддерживать выполнение дедлайна при наличии
*систематического смещения* в оценках длительности задач,
корректно учитывая неопределённость этого смещения.

Формально отслеживаемый показатель:
    P(finish_replanned ≤ deadline | прогресс проекта)

Эта характеристика составная:
    P = f(качество оценки смещения, распространение неопределённости,
          регуляризация на малых выборках)

════════════════════════════════════════════════════════════════
ТРИ СТРАТЕГИИ
════════════════════════════════════════════════════════════════

1. Static (базовый уровень)
   Нет адаптации к наблюдаемым данным.
   Постериор: θ ~ N(0, 0) — нулевое смещение, нулевая неопределённость.
   Всегда использует base_percentile без коррекции.

2. EVM (Earned Value Management — существующий аналог)
   Industry-standard подход: вычисляет среднеарифметический коэффициент
   отклонения (actual_i / mean_i) по выполненным задачам.
   Применяет коррекцию как точечную оценку, без учёта неопределённости.
   Постериор: θ = log(mean(actual_i/mean_i)), var = 0.
   Те же ограничения, что у EAC = BAC/SPI в PMBOK:
     — линейная экстраполяция (не логарифмическая)
     — нет prior regularization → перегрев на малых выборках
     — нет propagation of posterior uncertainty

3. Bayesian (ваша модель)
   Гауссовское сопряжённое обновление в лог-пространстве:
     ln(actual_i) = mu_log_i + θ + ε_i
     θ ~ N(prior_mean, prior_std²) → after n obs: N(post_mean, post_var)
   Advantages over EVM:
     (a) Prior shrinkage — регуляризует к "нет смещения" при малой выборке
     (b) Log-scale — корректно для логнормального распределения длительностей
     (c) Precision weighting — наблюдения с меньшим σ_i имеют больший вес
     (d) Posterior variance → задачи дополнительно консервативно буферизируются
         пропорционально неопределённости оценки смещения

════════════════════════════════════════════════════════════════
ВЫХОДНЫЕ ДАННЫЕ
════════════════════════════════════════════════════════════════
output/scheduling_comparison/
    results.csv            — сырые данные по каждому запуску
    headline.png           — основная панель: 3×3 сетка deadline_met_rate
    bias_error.png         — точность оценки смещения vs obs_fraction
    finish_dist.png        — распределение project_finish по трём методам
    convergence.png        — сходимость байеса, EVM, статики по ключевым сценариям
    percentile_used.png    — используемый персентиль (показывает адаптацию буфера)
    summary.xlsx           — сводная таблица

Запуск
------
    python -m experiments.scheduling_comparison
"""

from __future__ import annotations

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.scenario_generator import SCENARIOS, load_scheduled_tasks
from models.task import Task
from services.bayes_rescheduler import (
    GlobalBiasPosterior,
    estimate_global_bias_posterior,
    reconstruct_fact_schedule,
    build_rescheduled_plan,
    reschedule_with_fixed_project_deadline,
)
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from utils.lognormal import lognorm_log_params


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

TASK_FILE    = "data/tasks.csv"
OUTPUT_DIR   = Path("output/scheduling_comparison")

BASE_PERCENTILE       = 0.9
PLAN_SEED             = 42
OBSERVATION_FRACTIONS = [0.25, 0.50, 0.75]
N_SEEDS               = 8          # больше для стабильных оценок
MAX_ITER              = 20

# Сценарии, хорошо раскрывающие различия между методами
SCENARIOS_SUBSET = [
    "baseline",             # true_bias=1.0; чистый сигнал: нет смещения
    "global_overrun_1.5x",  # true_bias=1.5; умеренный глобальный перерасход
    "global_overrun_2x",    # true_bias=2.0; сильный перерасход
    "global_underrun",      # true_bias=0.67; экономия времени
    "high_variance",        # те же средние, но σ в 2×; тест propagation
    "role_bottleneck",      # смещение только разработчиков; partial signal
    "cascading_delay",      # нарастающее смещение; сигнал нелинейный
    "single_outlier",       # один выброс; тест robustness
    "deadline_impossible",  # bias=3×; стресс-тест
]

TRUE_BIAS = {
    "baseline":             1.00,
    "global_overrun_1.5x":  1.50,
    "global_overrun_2x":    2.00,
    "global_underrun":      0.67,
    "high_variance":        1.00,
    "role_bottleneck":      1.00,
    "cascading_delay":      1.50,
    "single_outlier":       1.00,
    "deadline_impossible":  3.00,
}

METHOD_COLORS = {
    "Static":   "#AAAAAA",
    "EVM":      "#F28C38",
    "Bayesian": "#4C9BE8",
}
METHOD_ORDER = ["Static", "EVM", "Bayesian"]


# ---------------------------------------------------------------------------
# EVM оценка смещения (точечная, в лог-пространстве)
# ---------------------------------------------------------------------------

def estimate_evm_posterior(
    tasks: list[Task],
    progress: dict[int, dict],
    prior_mean: float = 0.0,
    prior_std:  float = 0.30,
) -> tuple[GlobalBiasPosterior, int]:
    """
    EVM-style point estimate of systematic bias.

    Вычисляет среднеарифметическое отношение actual_i / mean_i,
    переводит в лог-пространство и возвращает GlobalBiasPosterior
    с нулевой дисперсией (точечная оценка — нет неопределённости).

    При отсутствии наблюдений — возвращает prior (как у Bayes без данных).
    """
    ratios = []
    for task in tasks:
        info = progress.get(task.task_id, {})
        if info.get("status") != "done":
            continue
        actual = float(info.get("actual_duration", 0))
        if actual > 0 and task.mean > 0:
            ratios.append(actual / task.mean)

    n = len(ratios)
    if n == 0:
        # Нет данных — используем prior (аналогично Bayes без наблюдений)
        return GlobalBiasPosterior(mean=prior_mean, var=prior_std ** 2), 0

    mean_ratio = float(np.mean(ratios))
    log_bias   = float(np.log(max(mean_ratio, 1e-9)))

    # var=0: EVM — точечная оценка, без учёта неопределённости
    return GlobalBiasPosterior(mean=log_bias, var=0.0), n


# ---------------------------------------------------------------------------
# Static: всегда prior без данных
# ---------------------------------------------------------------------------

def estimate_static_posterior() -> tuple[GlobalBiasPosterior, int]:
    """Static: не адаптируется, считает смещение нулевым (нет данных)."""
    return GlobalBiasPosterior(mean=0.0, var=0.0), 0


# ---------------------------------------------------------------------------
# Универсальный перепланировщик: принимает любой постериор
# ---------------------------------------------------------------------------

def _reschedule_with_posterior(
    tasks: list[Task],
    progress: dict[int, dict],
    posterior: GlobalBiasPosterior,
    target_finish: float,
    base_percentile: float = BASE_PERCENTILE,
    min_percentile:  float = 0.05,
    max_percentile:  float = 0.99,
    max_iter:        int   = MAX_ITER,
    tolerance:       float = 1e-3,
) -> dict:
    """
    Перепланирует проект с заданным постериором.

    Использует тот же бинарный поиск, что и reschedule_with_fixed_project_deadline,
    но принимает постериор извне (что позволяет подставлять EVM / Static / Bayes).
    """
    _, current_time = reconstruct_fact_schedule(tasks, progress)

    def _plan(p: float) -> tuple[list[Task], float]:
        t, f, _ = build_rescheduled_plan(tasks, progress, current_time, p, posterior)
        return t, f

    def _result(p, t, f, met, msg):
        return {
            "tasks":            t,
            "project_finish":   f,
            "current_time":     current_time,
            "used_percentile":  p,
            "deadline_met":     met,
            "posterior":        posterior,
            "bias_factor_mean": float(np.exp(posterior.mean)),
            "message":          msg,
        }

    base_tasks, base_finish = _plan(base_percentile)

    # Все задачи выполнены
    if abs(base_finish - current_time) < 1e-9:
        return _result(base_percentile, base_tasks, base_finish,
                       base_finish <= target_finish, "All done.")

    if base_finish <= target_finish:
        # Дедлайн выполняется при base_percentile
        if posterior.mean < -0.05:
            # Отрицательное смещение — ищем выше (рекупируем буфер)
            hi_tasks, hi_finish = _plan(max_percentile)
            if hi_finish <= target_finish:
                return _result(max_percentile, hi_tasks, hi_finish, True, "Negative bias; max p.")
            lo, hi, bt, bf, bp = base_percentile, max_percentile, base_tasks, base_finish, base_percentile
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                t, f = _plan(mid)
                if f <= target_finish:
                    bt, bf, bp = t, f, mid; lo = mid
                else:
                    hi = mid
                if hi - lo < tolerance:
                    break
            return _result(bp, bt, bf, True, "Negative bias upward.")
        return _result(base_percentile, base_tasks, base_finish, True, "OK.")

    # Дедлайн нарушается — ищем вниз
    min_tasks, min_finish = _plan(min_percentile)
    if min_finish > target_finish:
        return _result(min_percentile, min_tasks, min_finish, False, "Unachievable.")

    lo, hi = min_percentile, base_percentile
    bt, bf, bp = min_tasks, min_finish, min_percentile
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t, f = _plan(mid)
        if f <= target_finish:
            bt, bf, bp = t, f, mid; lo = mid
        else:
            hi = mid
        if hi - lo < tolerance:
            break
    return _result(bp, bt, bf, True, f"Lowered p to {bp:.3f}.")


# ---------------------------------------------------------------------------
# Один запуск
# ---------------------------------------------------------------------------

def run_single(
    scenario_name: str,
    n_observed:    int,
    seed:          int,
    scheduled_tasks: list[Task],
    target_finish: float,
    task_file:     str   = TASK_FILE,
    base_percentile: float = BASE_PERCENTILE,
) -> dict:
    scenario = SCENARIOS[scenario_name]
    n_tasks  = len(scheduled_tasks)

    progress = scenario["fn"](
        scheduled_tasks, n_observed=n_observed, seed=seed, **scenario["kwargs"]
    )

    results = {}
    for method in METHOD_ORDER:
        fresh = load_tasks_from_csv(task_file)
        build_schedule(fresh, percentile=base_percentile, seed=PLAN_SEED)

        t0 = time.perf_counter()
        try:
            if method == "Bayesian":
                res = reschedule_with_fixed_project_deadline(
                    tasks=fresh, progress=progress,
                    target_finish_time=target_finish,
                    base_percentile=base_percentile,
                )
                post, n_used = res["posterior"], res["used_observations"]

            elif method == "EVM":
                post, n_used = estimate_evm_posterior(fresh, progress)
                res = _reschedule_with_posterior(
                    fresh, progress, post, target_finish, base_percentile
                )
                res["used_observations"] = n_used

            else:  # Static
                post, n_used = estimate_static_posterior()
                res = _reschedule_with_posterior(
                    fresh, progress, post, target_finish, base_percentile
                )
                res["used_observations"] = n_used

            results[method] = {
                "finish":    round(res["project_finish"], 3),
                "met":       bool(res["deadline_met"]),
                "pct":       round(res["used_percentile"], 4),
                "bias_est":  round(float(np.exp(post.mean)), 4),
                "post_var":  round(float(post.var), 5),
                "n_used":    n_used,
                "time":      round(time.perf_counter() - t0, 3),
                "error":     None,
            }
        except Exception as exc:
            results[method] = {
                "finish": np.nan, "met": False, "pct": np.nan,
                "bias_est": np.nan, "post_var": np.nan, "n_used": 0,
                "time": round(time.perf_counter() - t0, 3), "error": str(exc),
            }

    true_bias = TRUE_BIAS.get(scenario_name, 1.0)

    row: dict = {
        "scenario":    scenario_name,
        "obs_fraction": round(n_observed / n_tasks, 3),
        "n_observed":  n_observed,
        "seed":        seed,
        "true_bias":   true_bias,
    }
    for method, r in results.items():
        pfx = method[:3].lower()  # bay / evm / sta
        for k, v in r.items():
            row[f"{pfx}_{k}"] = v

    # Bias estimation error (delta vs true bias)
    for method in METHOD_ORDER:
        pfx = method[:3].lower()
        est = row.get(f"{pfx}_bias_est", np.nan)
        row[f"{pfx}_bias_err"] = abs(est - true_bias) if not np.isnan(est) else np.nan

    return row


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def run_all(
    task_file:    str   = TASK_FILE,
    base_p:       float = BASE_PERCENTILE,
    plan_seed:    int   = PLAN_SEED,
    obs_fracs:    list  = OBSERVATION_FRACTIONS,
    n_seeds:      int   = N_SEEDS,
    scenarios:    list  = SCENARIOS_SUBSET,
    output_dir:   Path  = OUTPUT_DIR,
    verbose:      bool  = True,
) -> tuple[pd.DataFrame, float, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduled = load_scheduled_tasks(task_file, base_p, plan_seed)
    n_tasks   = len(scheduled)
    deadline  = max(t.planned_end_time for t in scheduled)
    obs_counts = [max(1, round(f * n_tasks)) for f in obs_fracs]
    total      = len(scenarios) * len(obs_counts) * n_seeds

    if verbose:
        print(f"Задач: {n_tasks} | Дедлайн: {deadline:.1f} | "
              f"Запусков: {total} ({len(scenarios)} сц × {len(obs_counts)} obs × {n_seeds} seeds)")
        print()

    rows = []
    done = 0
    t0   = time.perf_counter()

    for sc in scenarios:
        for n_obs in obs_counts:
            for seed in range(n_seeds):
                row = run_single(sc, n_obs, seed, scheduled, deadline,
                                 task_file, base_p)
                rows.append(row)
                done += 1
                if verbose:
                    eta = (time.perf_counter() - t0) / done * (total - done)
                    b = "Y" if row.get("bay_met") else "N"
                    e = "Y" if row.get("evm_met") else "N"
                    s = "Y" if row.get("sta_met") else "N"
                    imp = (row.get("bay_finish", 0) or 0) - (row.get("evm_finish", 0) or 0)
                    print(f"  [{done:3d}/{total}] {sc:<25} obs={n_obs:2d} s={seed} "
                          f"Sta={s} EVM={e} Bay={b}  "
                          f"Bayes-EVM={imp:+.1f}d  ETA {eta/60:.1f}m")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)
    if verbose:
        print(f"\nCSV -> {output_dir / 'results.csv'}")

    return df, deadline, n_tasks


# ---------------------------------------------------------------------------
# Вспомогательные функции для графиков
# ---------------------------------------------------------------------------

def _frac_label(frac: float) -> str:
    for ref, lbl in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
        if abs(frac - ref) < 0.06:
            return lbl
    return f"{frac:.0%}"


def _nearest_frac(df: pd.DataFrame, target: float) -> float:
    fracs = df["obs_fraction"].unique()
    return float(min(fracs, key=lambda f: abs(f - target)))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    mask = True
    for m in ["bay", "evm", "sta"]:
        mask = mask & df[f"{m}_error"].isna()
    return df[mask].copy()


def _met_col(method: str) -> str:
    return f"{method[:3].lower()}_met"

def _fin_col(method: str) -> str:
    return f"{method[:3].lower()}_finish"

def _pct_col(method: str) -> str:
    return f"{method[:3].lower()}_pct"

def _err_col(method: str) -> str:
    return f"{method[:3].lower()}_bias_err"


# ---------------------------------------------------------------------------
# ГЛАВНЫЙ ГРАФИК: 3×3 Deadline met rate
# ---------------------------------------------------------------------------

def plot_headline(df: pd.DataFrame, deadline: float, output_dir: Path) -> None:
    """
    Сетка сценарии × obs_fraction.
    Каждая ячейка — 3 вертикальных бара: Static / EVM / Bayesian.
    Высота = deadline_met_rate.
    """
    clean  = _clean(df)
    fracs  = sorted(clean["obs_fraction"].unique())
    scens  = [s for s in SCENARIOS_SUBSET if s in clean["scenario"].unique()]
    n_sc   = len(scens)
    n_fr   = len(fracs)

    fig_w = 4.5 * n_fr
    fig_h = 2.0 * n_sc + 1.5
    fig, axes = plt.subplots(n_sc, n_fr,
                              figsize=(fig_w, fig_h),
                              sharex=False, sharey=True)

    if n_sc == 1:
        axes = [axes]
    if n_fr == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(
        "Надёжность выполнения дедлайна (deadline_met_rate)\n"
        "Серый = Static | Оранжевый = EVM | Синий = Bayesian",
        fontsize=12, fontweight="bold",
    )

    bw = 0.25
    x  = np.arange(3)

    for ri, sc in enumerate(scens):
        for ci, frac in enumerate(fracs):
            ax  = axes[ri][ci]
            sub = clean[(clean["scenario"] == sc) &
                        ((clean["obs_fraction"] - frac).abs() < 0.06)]

            rates = [sub[_met_col(m)].mean() if len(sub) else 0 for m in METHOD_ORDER]
            colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

            bars = ax.bar(x, rates, width=0.65, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            ax.axhline(0.8, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_ylim(0, 1.15)
            ax.set_xticks(x)
            ax.set_xticklabels(["Sta", "EVM", "Bay"], fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            for bar, rate in zip(bars, rates):
                if rate > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            rate + 0.03, f"{rate:.0%}",
                            ha="center", fontsize=6.5, fontweight="bold")

            if ri == 0:
                ax.set_title(f"obs={_frac_label(frac)}", fontsize=9)
            if ci == 0:
                short = sc.replace("global_", "").replace("_", "\n")
                ax.set_ylabel(short, fontsize=7.5, rotation=0,
                              ha="right", va="center", labelpad=40)

    plt.tight_layout()
    path = output_dir / "headline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  headline -> {path}")


# ---------------------------------------------------------------------------
# ГРАФИК 2: Bias estimation error vs obs_fraction
# ---------------------------------------------------------------------------

def plot_bias_error(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Ошибка оценки смещения |est_bias - true_bias| vs observation fraction.
    Три кривые на одном графике (усреднено по сценариям и семенам).
    """
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())
    scens = [s for s in SCENARIOS_SUBSET if s in clean["scenario"].unique()]

    # Один большой график + маленькие по ключевым сценариям
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Ошибка оценки смещения |bias_estimated - true_bias|\n"
        "Меньше = точнее. Ключевое отличие Bayesian от EVM при малом числе наблюдений.",
        fontsize=10, fontweight="bold",
    )

    highlight_scens = ["global_overrun_1.5x", "high_variance", "single_outlier"]

    for ax, sc in zip(axes, highlight_scens):
        sub_sc = clean[clean["scenario"] == sc]
        true_b = TRUE_BIAS.get(sc, 1.0)

        for method in METHOD_ORDER:
            col    = _err_col(method)
            means  = []
            stds   = []
            fr_vals = []
            for frac in fracs:
                sub_f = sub_sc[(sub_sc["obs_fraction"] - frac).abs() < 0.06]
                vals  = sub_f[col].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    fr_vals.append(frac * 100)

            if means:
                ax.plot(fr_vals, means, "-o", color=METHOD_COLORS[method],
                        linewidth=2.0, markersize=6, label=method)
                ax.fill_between(fr_vals,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                color=METHOD_COLORS[method], alpha=0.12)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Выполнено задач (%)", fontsize=9)
        ax.set_ylabel("|bias_est - true_bias|", fontsize=9)
        ax.set_title(f"Сценарий: {sc}\ntrue_bias={true_b}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = output_dir / "bias_error.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  bias_error -> {path}")


# ---------------------------------------------------------------------------
# ГРАФИК 3: Распределение project_finish по методам
# ---------------------------------------------------------------------------

def plot_finish_distribution(df: pd.DataFrame, deadline: float,
                              output_dir: Path) -> None:
    """
    Violin plots: распределение project_finish для каждого метода.
    Панели = obs_fraction.  Строки = ключевые сценарии.
    """
    clean  = _clean(df)
    fracs  = sorted(clean["obs_fraction"].unique())
    scens  = ["global_overrun_1.5x", "high_variance",
               "cascading_delay", "global_underrun"]

    fig, axes = plt.subplots(len(scens), len(fracs),
                              figsize=(5 * len(fracs), 3 * len(scens) + 1),
                              sharey="row")
    fig.suptitle(
        "Распределение project_finish по трём стратегиям\n"
        "Красная пунктирная линия = дедлайн",
        fontsize=11, fontweight="bold",
    )

    for ri, sc in enumerate(scens):
        for ci, frac in enumerate(fracs):
            ax  = axes[ri][ci]
            sub = clean[(clean["scenario"] == sc) &
                        ((clean["obs_fraction"] - frac).abs() < 0.06)]

            positions = [0, 1, 2]
            data = [sub[_fin_col(m)].dropna().values for m in METHOD_ORDER]
            colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

            vp = ax.violinplot(
                [d if len(d) > 1 else [deadline] for d in data],
                positions=positions,
                showmedians=True, showextrema=True,
            )
            for body, color in zip(vp["bodies"], colors):
                body.set_facecolor(color)
                body.set_alpha(0.65)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

            ax.axhline(deadline, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.8)
            ax.set_xticks(positions)
            ax.set_xticklabels(["Sta", "EVM", "Bay"], fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            if ri == 0:
                ax.set_title(f"obs={_frac_label(frac)}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(sc.replace("global_", "").replace("_", " "),
                               fontsize=8)

    plt.tight_layout()
    path = output_dir / "finish_dist.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  finish_dist -> {path}")


# ---------------------------------------------------------------------------
# ГРАФИК 4: Используемый персентиль (буфер)
# ---------------------------------------------------------------------------

def plot_percentile_adaptation(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Средний используемый персентиль vs obs_fraction по сценариям.
    Демонстрирует адаптацию буфера.
    """
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())
    scens = [s for s in SCENARIOS_SUBSET if s in clean["scenario"].unique()]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle(
        "Адаптация планового буфера (используемый персентиль)\n"
        "Bayesian динамически регулирует консервативность; "
        "Static фиксирован; EVM — резкие скачки",
        fontsize=11, fontweight="bold",
    )

    for ax, sc in zip(axes.flat, scens):
        sub_sc = clean[clean["scenario"] == sc]
        fr_pct = [f * 100 for f in fracs]

        for method in METHOD_ORDER:
            col   = _pct_col(method)
            means = []
            stds  = []
            for frac in fracs:
                sub_f = sub_sc[(sub_sc["obs_fraction"] - frac).abs() < 0.06]
                vals  = sub_f[col].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std()   if len(vals) > 1 else 0)

            ax.plot(fr_pct, means, "-o", color=METHOD_COLORS[method],
                    linewidth=2, markersize=6, label=method)
            ax.fill_between(fr_pct,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=METHOD_COLORS[method], alpha=0.1)

        ax.axhline(BASE_PERCENTILE, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5, label="base")
        ax.set_title(sc.replace("global_", "").replace("_", " "), fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower right", ncol=2)

    for ax in axes[-1]:
        ax.set_xlabel("Выполнено задач (%)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Персентиль", fontsize=9)

    plt.tight_layout()
    path = output_dir / "percentile_used.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  percentile_used -> {path}")


# ---------------------------------------------------------------------------
# ГРАФИК 5: Сводная панель преимуществ
# ---------------------------------------------------------------------------

def plot_advantage_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Для каждого метода: среднее улучшение deadline_met_rate
    Bayesian - Static  и  Bayesian - EVM  по всем наблюдениям.
    Показывает, где вклад адаптации максимален.
    """
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())
    scens = [s for s in SCENARIOS_SUBSET if s in clean["scenario"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(scens) * 0.55 + 1.5)),
                              sharey=True)
    fig.suptitle(
        "Прирост надёжности дедлайна от применения адаптивных стратегий\n"
        "Левая панель: Bayesian - Static  |  Правая: Bayesian - EVM",
        fontsize=11, fontweight="bold",
    )

    y = np.arange(len(scens))
    ylabels = [s.replace("global_", "g_").replace("_", " ") for s in scens]

    for ax, (ref_method, ref_col, title) in zip(axes, [
        ("Static", "sta_met", "Bayesian - Static\n(ценность ЛЮБОЙ адаптации)"),
        ("EVM",    "evm_met", "Bayesian - EVM\n(ценность байесовской оценки vs EVM)"),
    ]):
        means_by_frac = {}
        stds_by_frac  = {}
        for frac in fracs:
            sub = clean[(clean["obs_fraction"] - frac).abs() < 0.06]
            vals = sub.groupby("scenario").apply(
                lambda g: g["bay_met"].mean() - g[ref_col].mean()
            )
            means_by_frac[frac] = {sc: vals.get(sc, 0) for sc in scens}

        cmap = plt.get_cmap("cool")
        for fi, frac in enumerate(fracs):
            vals   = [means_by_frac[frac].get(sc, 0) for sc in scens]
            offset = (fi - len(fracs) / 2) * 0.24
            colors = ["#2ecc71" if v > 0.05 else
                      "#e74c3c" if v < -0.05 else
                      "#AAAAAA" for v in vals]
            ax.barh(y + offset, vals, height=0.22, color=colors, alpha=0.8,
                    label=_frac_label(frac), edgecolor="white")

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Δ deadline_met_rate", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.legend(title="obs", fontsize=8, loc="lower right")

    plt.tight_layout()
    path = output_dir / "advantage_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  advantage_summary -> {path}")


# ---------------------------------------------------------------------------
# Текстовый отчёт
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, deadline: float) -> None:
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())

    print()
    print("=" * 70)
    print("ИТОГИ: Static vs EVM vs Bayesian")
    print("=" * 70)
    print(f"Дедлайн: {deadline:.1f} дней\n")

    for frac in fracs:
        sub = clean[(clean["obs_fraction"] - frac).abs() < 0.06]
        print(f"--- Наблюдение {_frac_label(frac)} ({len(sub)} запусков) ---")
        for m in METHOD_ORDER:
            mc = _met_col(m)
            print(f"  {m:<10} deadline_met: {sub[mc].mean():.1%}  "
                  f"mean_bias_err: {sub[_err_col(m)].mean():.3f}  "
                  f"mean_pct: {sub[_pct_col(m)].mean():.3f}")
        # Key gaps
        bay_rate = sub["bay_met"].mean()
        evm_rate = sub["evm_met"].mean()
        sta_rate = sub["sta_met"].mean()
        print(f"  -> Bayesian над EVM:    {bay_rate - evm_rate:+.1%}")
        print(f"  -> Bayesian над Static: {bay_rate - sta_rate:+.1%}")
        print()

    # По сценариям при obs~50%
    mid_frac = _nearest_frac(clean, 0.50)
    mid = clean[(clean["obs_fraction"] - mid_frac).abs() < 0.06]
    print(f"\nПо сценариям (obs~{_frac_label(mid_frac)}):")
    rows = []
    for sc in SCENARIOS_SUBSET:
        sub = mid[mid["scenario"] == sc]
        if sub.empty:
            continue
        rows.append({
            "scenario":       sc,
            "true_bias":      TRUE_BIAS.get(sc, "?"),
            "Static_met":     f"{sub['sta_met'].mean():.0%}",
            "EVM_met":        f"{sub['evm_met'].mean():.0%}",
            "Bayes_met":      f"{sub['bay_met'].mean():.0%}",
            "Bay-EVM":        f"{sub['bay_met'].mean() - sub['evm_met'].mean():+.0%}",
            "Bay-Sta":        f"{sub['bay_met'].mean() - sub['sta_met'].mean():+.0%}",
            "EVM_bias_err":   f"{sub['evm_bias_err'].mean():.3f}",
            "Bay_bias_err":   f"{sub['bay_bias_err'].mean():.3f}",
        })
    print(pd.DataFrame(rows).set_index("scenario").to_string())
    print("=" * 70)


# ---------------------------------------------------------------------------
# Excel-сводка
# ---------------------------------------------------------------------------

def export_excel(df: pd.DataFrame, deadline: float, output_dir: Path) -> None:
    clean = _clean(df)
    fracs = sorted(clean["obs_fraction"].unique())

    rows = []
    for sc in SCENARIOS_SUBSET:
        for frac in fracs:
            sub = clean[(clean["scenario"] == sc) &
                        ((clean["obs_fraction"] - frac).abs() < 0.06)]
            if sub.empty:
                continue
            row = {
                "scenario": sc,
                "true_bias": TRUE_BIAS.get(sc, "?"),
                "obs_frac": _frac_label(frac),
                "n": len(sub),
            }
            for m in METHOD_ORDER:
                pfx = m[:3].lower()
                row[f"{m}_met"]      = round(sub[f"{pfx}_met"].mean(), 3)
                row[f"{m}_finish"]   = round(sub[f"{pfx}_finish"].mean(), 2)
                row[f"{m}_pct"]      = round(sub[f"{pfx}_pct"].mean(), 3)
                row[f"{m}_bias_err"] = round(sub[f"{pfx}_bias_err"].mean(), 4)
                row[f"{m}_time"]     = round(sub[f"{pfx}_time"].mean(), 3)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    xlsx_path  = output_dir / "summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Raw", index=False)

    print(f"  Excel -> {xlsx_path}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  default=TASK_FILE)
    parser.add_argument("--out",    default=str(OUTPUT_DIR))
    parser.add_argument("--seeds",  default=N_SEEDS, type=int)
    args = parser.parse_args()

    out_dir = Path(args.out)

    print()
    print("=" * 70)
    print("УПРАВЛЕНЧЕСКАЯ ХАРАКТЕРИСТИКА: Calibrated Schedule Buffer Reliability")
    print("Сравнение: Static | EVM | Bayesian (ваша модель)")
    print("=" * 70)

    df, deadline, n_tasks = run_all(
        task_file=args.tasks,
        n_seeds=args.seeds,
        output_dir=out_dir,
        verbose=True,
    )

    print("\nСтроим графики...")
    plot_headline(df, deadline, out_dir)
    plot_bias_error(df, out_dir)
    plot_finish_distribution(df, deadline, out_dir)
    plot_percentile_adaptation(df, out_dir)
    plot_advantage_summary(df, out_dir)
    export_excel(df, deadline, out_dir)

    print_report(df, deadline)
