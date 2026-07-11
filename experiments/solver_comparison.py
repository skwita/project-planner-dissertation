"""
Сравнение планировщиков: OR-Tools CP-SAT (дефолтный решатель) vs жадная модель.

Оба решателя работают с одинаковыми входными данными — средними (mean) длительностями
задач из tasks.csv и одинаковыми ограничениями:
  - Зависимости между задачами (предшествование)
  - Ресурсные ограничения: не более 1 задачи одновременно для каждой роли

OR-Tools CP-SAT минимизирует makespan (длину всего проекта).
Жадная модель (services/scheduler.py) выполняет топологическую сортировку (BFS/Kahn)
и последовательно назначает задачи в пределах каждой роли.

Выходные данные
---------------
    output/solver_comparison/
      gantt_comparison.png      — диаграммы Ганта обоих решателей
      deviation_start.png       — отклонения времён начала по задачам
      deviation_end.png         — отклонения времён окончания по задачам
      deviation_summary.png     — сводные боксплоты по ролям
      comparison_table.csv      — подробная таблица отклонений

Использование
-------------
    python -m experiments.solver_comparison
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 console output on Windows (avoids cp1251 UnicodeEncodeError)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.parser import load_tasks_from_csv
from services.scheduler import topo_sort_tasks
from models.task import Task


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

TASK_FILE  = "data/tasks.csv"
OUTPUT_DIR = Path("output/solver_comparison")
SCALE      = 100     # множитель для перевода float -> int в CP-SAT
SOLVER_TIMEOUT_SEC = 60.0

ROLE_COLORS = {
    "аналитик":    "#4C9BE8",   # синий
    "разработчик": "#F28C38",   # оранжевый
    "тестировщик": "#5DBB63",   # зелёный
}
DEFAULT_COLOR = "#AAAAAA"


# ---------------------------------------------------------------------------
# Жадный планировщик (ваша модель) — mean-длительности
# ---------------------------------------------------------------------------

def build_greedy_mean_schedule(tasks: list[Task]) -> list[Task]:
    """
    Запускает топологический планировщик (services/scheduler.py logic)
    с planned_duration = task.mean для каждой задачи.

    Изменяет объекты Task на месте: заполняет planned_start_time,
    planned_end_time, planned_duration.

    Returns:
        Тот же список задач с проставленными временами.
    """
    tasks = deepcopy(tasks)
    for task in tasks:
        task.planned_duration = task.mean

    task_map: dict[int, Task] = {t.task_id: t for t in tasks}
    order = topo_sort_tasks(tasks)

    role_ready: dict[str, float] = defaultdict(float)

    for tid in order:
        task = task_map[tid]
        dep_end = max(
            (task_map[dep].planned_end_time for dep in task.dependencies),
            default=0.0,
        )
        task.planned_start_time = max(dep_end, role_ready[task.role])
        task.planned_end_time   = task.planned_start_time + task.planned_duration
        role_ready[task.role]   = task.planned_end_time

    return tasks


# ---------------------------------------------------------------------------
# OR-Tools CP-SAT решатель
# ---------------------------------------------------------------------------

def build_ortools_schedule(
    tasks: list[Task],
    timeout: float = SOLVER_TIMEOUT_SEC,
    scale: int = SCALE,
) -> tuple[dict[int, float], dict[int, float], float, str]:
    """
    Находит оптимальный расписание, минимизирующее makespan, через CP-SAT.

    Модель:
      - Переменные: start_i ∈ [0, horizon], end_i = start_i + duration_i
      - Ограничения предшествования: end_j <= start_i для каждого dep j->i
      - Ресурсные ограничения: AddNoOverlap для всех задач одной роли
      - Цель: minimize max(end_i)

    Длительности переводятся в целые числа умножением на ``scale``
    (точность = 1/scale дня). После решения результаты делятся на ``scale``.

    Args:
        tasks:   Список задач (только читаются).
        timeout: Максимальное время решения в секундах.
        scale:   Множитель целочисленного масштабирования.

    Returns:
        (starts_float, ends_float, makespan_float, status_str)
        При неудаче возвращает пустые словари и makespan=inf.
    """
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers  = 8

    # Целочисленные длительности
    dur_int: dict[int, int] = {
        t.task_id: max(1, round(t.mean * scale))
        for t in tasks
    }

    # Горизонт: сумма всех длительностей — грубая верхняя оценка makespan
    horizon = sum(dur_int.values())

    # Создаём переменные
    starts_var:    dict[int, cp_model.IntVar] = {}
    ends_var:      dict[int, cp_model.IntVar] = {}
    intervals_var: dict[int, cp_model.IntervalVar] = {}

    for t in tasks:
        tid = t.task_id
        s   = model.new_int_var(0, horizon, f"s_{tid}")
        e   = model.new_int_var(0, horizon, f"e_{tid}")
        iv  = model.new_interval_var(s, dur_int[tid], e, f"iv_{tid}")
        starts_var[tid]    = s
        ends_var[tid]      = e
        intervals_var[tid] = iv

    # Ограничения предшествования
    for t in tasks:
        for dep_id in t.dependencies:
            model.add(starts_var[t.task_id] >= ends_var[dep_id])

    # Ресурсные ограничения: без перекрытий внутри роли
    role_ivals: dict[str, list] = defaultdict(list)
    for t in tasks:
        role_ivals[t.role].append(intervals_var[t.task_id])

    for role, ivals in role_ivals.items():
        if len(ivals) > 1:
            model.add_no_overlap(ivals)

    # Цель: минимизация makespan
    makespan_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan_var, list(ends_var.values()))
    model.minimize(makespan_var)

    # Решение
    status = solver.solve(model)
    status_str = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"[WARN] CP-SAT solver returned: {status_str}")
        return {}, {}, float("inf"), status_str

    starts_float = {tid: solver.value(s) / scale for tid, s in starts_var.items()}
    ends_float   = {tid: solver.value(e) / scale for tid, e in ends_var.items()}
    makespan_f   = solver.value(makespan_var) / scale

    return starts_float, ends_float, makespan_f, status_str


# ---------------------------------------------------------------------------
# Сравнительная таблица
# ---------------------------------------------------------------------------

def build_comparison_df(
    tasks: list[Task],
    greedy_tasks: list[Task],
    opt_starts: dict[int, float],
    opt_ends: dict[int, float],
) -> pd.DataFrame:
    """
    Собирает сводную таблицу отклонений жадного решателя от оптимального.

    Columns:
        task_id, role, mean, stddev,
        greedy_start, greedy_end,
        opt_start,    opt_end,
        delta_start,  delta_end   (greedy − optimal; > 0 означает задержку)
    """
    greedy_map = {t.task_id: t for t in greedy_tasks}

    rows = []
    for t in tasks:
        g  = greedy_map[t.task_id]
        gs = g.planned_start_time
        ge = g.planned_end_time
        os = opt_starts[t.task_id]
        oe = opt_ends[t.task_id]
        rows.append({
            "task_id":      t.task_id,
            "role":         t.role,
            "mean":         t.mean,
            "stddev":       t.stddev,
            "greedy_start": round(gs, 4),
            "greedy_end":   round(ge, 4),
            "opt_start":    round(os, 4),
            "opt_end":      round(oe, 4),
            "delta_start":  round(gs - os, 4),
            "delta_end":    round(ge - oe, 4),
        })

    return pd.DataFrame(rows).set_index("task_id")


# ---------------------------------------------------------------------------
# Диаграммы Ганта (сравнение)
# ---------------------------------------------------------------------------

def plot_gantt_comparison(
    tasks: list[Task],
    greedy_tasks: list[Task],
    opt_starts: dict[int, float],
    opt_ends: dict[int, float],
    opt_makespan: float,
    greedy_makespan: float,
    save_path: str,
) -> None:
    """
    Двойная диаграмма Ганта: слева — жадная модель, справа — CP-SAT.

    Задачи упорядочены по opt_start для удобства сравнения.
    Цвет полос = роль.
    """
    greedy_map = {t.task_id: t for t in greedy_tasks}
    task_map   = {t.task_id: t for t in tasks}

    sorted_ids = sorted(tasks, key=lambda t: opt_starts[t.task_id])
    y_labels   = [f"#{t.task_id} {t.role[:3]}" for t in sorted_ids]
    n_tasks    = len(sorted_ids)
    yticks     = list(range(n_tasks))

    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, n_tasks * 0.35)),
                             sharey=True)

    for ax, label, use_opt in zip(axes, ["Жадная модель (ваша)", "OR-Tools CP-SAT (оптимальная)"], [False, True]):
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Время (дни)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for yi, task in enumerate(sorted_ids):
            tid   = task.task_id
            role  = task_map[tid].role
            color = ROLE_COLORS.get(role, DEFAULT_COLOR)
            dur   = task_map[tid].mean

            if use_opt:
                start = opt_starts[tid]
            else:
                start = greedy_map[tid].planned_start_time

            ax.barh(yi, dur, left=start, height=0.6, color=color,
                    alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.text(start + dur / 2, yi, str(tid),
                    ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold")

        # Вертикальная линия makespan
        mk = opt_makespan if use_opt else greedy_makespan
        ax.axvline(mk, color="red", linestyle="--", linewidth=1.5,
                   label=f"Makespan = {mk:.1f} дн.")
        ax.legend(fontsize=9, loc="lower right")

    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(y_labels, fontsize=7)
    axes[0].invert_yaxis()

    # Легенда ролей
    legend_handles = [
        mpatches.Patch(color=c, label=r)
        for r, c in ROLE_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        f"Сравнение планировщиков | Жадная: {greedy_makespan:.1f} дн. "
        f"│ OR-Tools: {opt_makespan:.1f} дн. "
        f"│ Экономия: {greedy_makespan - opt_makespan:.1f} дн. "
        f"({(greedy_makespan - opt_makespan) / opt_makespan * 100:.1f}%)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gantt -> {save_path}")


# ---------------------------------------------------------------------------
# Диаграммы отклонений
# ---------------------------------------------------------------------------

def plot_deviations(
    df: pd.DataFrame,
    tasks: list[Task],
    save_start: str,
    save_end: str,
    save_summary: str,
) -> None:
    """
    Три графика отклонений жадного решателя от оптимального.

    1. Горизонтальный bar chart: delta_start по каждой задаче.
    2. Горизонтальный bar chart: delta_end по каждой задаче.
    3. Сводные боксплоты delta_start / delta_end по ролям.
    """
    task_map   = {t.task_id: t for t in tasks}
    sorted_ids = sorted(df.index, key=lambda tid: df.loc[tid, "opt_start"])
    roles      = df.loc[sorted_ids, "role"]

    bar_colors = [ROLE_COLORS.get(r, DEFAULT_COLOR) for r in roles]

    # ── 1. Отклонение времён начала ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_ids) * 0.28)))
    values  = df.loc[sorted_ids, "delta_start"].values
    ylabels = [f"#{tid} {df.loc[tid, 'role'][:3]}" for tid in sorted_ids]
    ypos    = np.arange(len(sorted_ids))

    ax.barh(ypos, values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ_начало = Жадная − Оптимальная (дни)", fontsize=10)
    ax.set_title("Отклонение времён начала задач\n"
                 "(> 0 = жадная начинает позже оптимальной)", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Аннотации значимых отклонений
    for y, v in zip(ypos, values):
        if abs(v) >= 1.0:
            ax.text(v + (0.1 if v >= 0 else -0.1), y, f"{v:.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=6.5)

    legend_handles = [mpatches.Patch(color=c, label=r)
                      for r, c in ROLE_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_start, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  delta_start -> {save_start}")

    # ── 2. Отклонение времён окончания ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_ids) * 0.28)))
    values  = df.loc[sorted_ids, "delta_end"].values

    ax.barh(ypos, values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ_конец = Жадная − Оптимальная (дни)", fontsize=10)
    ax.set_title("Отклонение времён окончания задач\n"
                 "(> 0 = жадная заканчивает позже оптимальной)", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for y, v in zip(ypos, values):
        if abs(v) >= 1.0:
            ax.text(v + (0.1 if v >= 0 else -0.1), y, f"{v:.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=6.5)

    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_end, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  delta_end -> {save_end}")

    # ── 3. Сводные боксплоты по ролям ───────────────────────────────────
    unique_roles = sorted(df["role"].unique())
    n_roles      = len(unique_roles)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_roles * 1.2 + 2)))
    fig.suptitle("Отклонения жадной модели от оптимальной: распределение по ролям",
                 fontsize=11, fontweight="bold")

    for ax, col, xlabel in zip(
        axes,
        ["delta_start", "delta_end"],
        ["Δ начало (дни)", "Δ конец (дни)"],
    ):
        data   = [df[df["role"] == r][col].values for r in unique_roles]
        bp     = ax.boxplot(data, vert=False, tick_labels=unique_roles,
                            patch_artist=True, notch=False)
        for patch, role in zip(bp["boxes"], unique_roles):
            patch.set_facecolor(ROLE_COLORS.get(role, DEFAULT_COLOR))
            patch.set_alpha(0.8)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"Распределение {col}", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_summary, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  summary -> {save_summary}")


# ---------------------------------------------------------------------------
# Дополнительный график: scatter (opt_start vs greedy_start)
# ---------------------------------------------------------------------------

def plot_start_scatter(
    df: pd.DataFrame,
    greedy_makespan: float,
    opt_makespan: float,
    save_path: str,
) -> None:
    """
    Scatter: оптимальное время начала (ось X) vs жадное (ось Y).

    Точки на диагонали — задачи, стартующие одновременно.
    Точки выше диагонали — жадная назначает позже.
    Цвет точек — роль.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for role, color in ROLE_COLORS.items():
        mask = df["role"] == role
        ax.scatter(
            df.loc[mask, "opt_start"],
            df.loc[mask, "greedy_start"],
            c=color, label=role, s=60, edgecolors="white", linewidths=0.5,
            zorder=3,
        )
        for tid in df[mask].index:
            ax.annotate(
                str(tid),
                (df.loc[tid, "opt_start"], df.loc[tid, "greedy_start"]),
                fontsize=6, color="grey", xytext=(3, 2),
                textcoords="offset points",
            )

    # Диагональ y = x
    lim = max(greedy_makespan, opt_makespan) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="y = x")

    ax.axvline(opt_makespan,    color="red",    linestyle=":",
               linewidth=1.2, label=f"Opt makespan {opt_makespan:.1f}")
    ax.axhline(greedy_makespan, color="orange", linestyle=":",
               linewidth=1.2, label=f"Жадный makespan {greedy_makespan:.1f}")

    ax.set_xlabel("Оптимальное время начала (CP-SAT), дни", fontsize=10)
    ax.set_ylabel("Жадное время начала (ваша модель), дни", fontsize=10)
    ax.set_title("Scatter: жадные vs оптимальные времена начала\n"
                 "(точки выше диагонали = жадный запускает задачу позже)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  scatter -> {save_path}")


# ---------------------------------------------------------------------------
# Текстовый отчёт
# ---------------------------------------------------------------------------

def print_summary(
    df: pd.DataFrame,
    greedy_makespan: float,
    opt_makespan: float,
    status_str: str,
) -> None:
    """Выводит краткое резюме в stdout."""
    savings    = greedy_makespan - opt_makespan
    savings_pc = savings / opt_makespan * 100

    print()
    print("=" * 60)
    print("СРАВНЕНИЕ РЕШАТЕЛЕЙ (задачи с mean-длительностями)")
    print("=" * 60)
    print(f"  Статус OR-Tools:             {status_str}")
    print(f"  Makespan жадной модели:      {greedy_makespan:.2f} дней")
    print(f"  Makespan OR-Tools CP-SAT:    {opt_makespan:.2f} дней")
    print(f"  Превышение жадной:           {savings:.2f} дней (+{savings_pc:.1f}%)")
    print()
    print("Отклонения delta_start (жадная − оптимальная):")
    print(f"  Среднее:   {df['delta_start'].mean():+.3f} дн.")
    print(f"  Медиана:   {df['delta_start'].median():+.3f} дн.")
    print(f"  Максимум:  {df['delta_start'].max():+.3f} дн.  (задача #{df['delta_start'].idxmax()})")
    print(f"  Задач с Δ > 0:  {(df['delta_start'] > 0.01).sum()}")
    print(f"  Задач с Δ = 0:  {(df['delta_start'].abs() <= 0.01).sum()}")
    print()
    print("Отклонения delta_end (жадная − оптимальная):")
    print(f"  Среднее:   {df['delta_end'].mean():+.3f} дн.")
    print(f"  Медиана:   {df['delta_end'].median():+.3f} дн.")
    print(f"  Максимум:  {df['delta_end'].max():+.3f} дн.  (задача #{df['delta_end'].idxmax()})")
    print()

    print("По ролям (среднее delta_start):")
    for role in sorted(df["role"].unique()):
        sub = df[df["role"] == role]
        print(f"  {role:<14}  n={len(sub):2d}  "
              f"mean_Δstart={sub['delta_start'].mean():+.2f} дн.  "
              f"mean_Δend={sub['delta_end'].mean():+.2f} дн.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def run_comparison(
    task_file: str = TASK_FILE,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Запускает полное сравнение жадного планировщика и OR-Tools CP-SAT.

    Returns:
        DataFrame с отклонениями по каждой задаче.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nЗагружаем задачи из: {task_file}")
    tasks = load_tasks_from_csv(task_file)
    if verbose:
        print(f"  Задач: {len(tasks)}, ролей: {len({t.role for t in tasks})}")

    # ── 1. Жадная модель ────────────────────────────────────────────────
    if verbose:
        print("\n[1/2] Жадный топологический планировщик...")
    greedy_tasks    = build_greedy_mean_schedule(tasks)
    greedy_makespan = max(t.planned_end_time for t in greedy_tasks)
    if verbose:
        print(f"  Makespan жадной: {greedy_makespan:.2f} дней")

    # ── 2. OR-Tools CP-SAT ──────────────────────────────────────────────
    if verbose:
        print(f"\n[2/2] OR-Tools CP-SAT (timeout={SOLVER_TIMEOUT_SEC}s)...")
    opt_starts, opt_ends, opt_makespan, status_str = build_ortools_schedule(tasks)
    if verbose:
        print(f"  Статус: {status_str}")
        print(f"  Makespan оптимального: {opt_makespan:.2f} дней")

    if opt_makespan == float("inf"):
        print("[ERROR] CP-SAT не нашёл решения. Завершаем.")
        return pd.DataFrame()

    # ── 3. Таблица отклонений ───────────────────────────────────────────
    df = build_comparison_df(tasks, greedy_tasks, opt_starts, opt_ends)

    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path)
    if verbose:
        print(f"\n  CSV -> {csv_path}")

    # ── 4. Диаграммы Ганта ──────────────────────────────────────────────
    if verbose:
        print("\nСтроим диаграммы...")
    plot_gantt_comparison(
        tasks, greedy_tasks, opt_starts, opt_ends,
        opt_makespan, greedy_makespan,
        str(output_dir / "gantt_comparison.png"),
    )

    # ── 5. Диаграммы отклонений ─────────────────────────────────────────
    plot_deviations(
        df, tasks,
        save_start   = str(output_dir / "deviation_start.png"),
        save_end     = str(output_dir / "deviation_end.png"),
        save_summary = str(output_dir / "deviation_summary.png"),
    )

    # ── 6. Scatter ──────────────────────────────────────────────────────
    plot_start_scatter(
        df, greedy_makespan, opt_makespan,
        str(output_dir / "scatter_starts.png"),
    )

    # ── 7. Текстовый отчёт ──────────────────────────────────────────────
    print_summary(df, greedy_makespan, opt_makespan, status_str)

    return df


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare greedy scheduler vs OR-Tools CP-SAT"
    )
    parser.add_argument(
        "--tasks",
        default=TASK_FILE,
        help=f"Path to tasks CSV (default: {TASK_FILE})",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    df = run_comparison(task_file=args.tasks, output_dir=Path(args.out), verbose=True)

    if not df.empty:
        print("\nТоп-10 задач по отклонению времени начала:")
        top = (df.sort_values("delta_start", ascending=False)
                 .head(10)[["role", "mean", "greedy_start", "opt_start", "delta_start"]])
        print(top.to_string())
