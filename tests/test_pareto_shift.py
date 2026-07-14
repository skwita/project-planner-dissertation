"""Tests for the Pareto-shift trajectory: data collection and plotting."""

import copy

import numpy as np
import pytest

from main import (
    _enforce_pareto_monotonicity,
    build_pareto_shift_updates,
    compute_pareto_idle_duration_curve,
    part5_pareto_shift_trajectory,
)
from services.bayes_rescheduler import reschedule_with_fixed_project_deadline
from services.parser import load_tasks_from_csv
from services.scheduler import build_schedule
from visualization.plot_idle_vs_duration import plot_pareto_shift_trajectory


def test_compute_pareto_idle_duration_curve_shapes(tiny_task_file):
    percentiles = [0.3, 0.6, 0.9]
    durations, idles = compute_pareto_idle_duration_curve(
        percentiles, task_file=tiny_task_file, seed=1, n_iter=200,
    )
    assert len(durations) == len(percentiles)
    assert len(idles) == len(percentiles)
    assert all(d > 0 for d in durations)
    assert all(i >= 0 for i in idles)


def test_compute_pareto_idle_duration_curve_is_monotonic_despite_mc_noise():
    """Regression test for the "mountain" bug: the true relationship is
    exactly monotonic (higher percentile -> longer or equal duration,
    shorter or equal idle), so with unseeded, moderate-n_iter runs (noisy
    enough to previously produce local up-then-down wiggles once sorted by
    duration for plotting) the *returned* curve must still come out clean.
    """
    percentiles = list(np.arange(0.05, 0.96, 0.05).tolist())
    durations, idles = compute_pareto_idle_duration_curve(
        percentiles, task_file="data/tasks.csv", seed=None, n_iter=1500,
    )

    durs = np.asarray(durations)
    idl = np.asarray(idles)
    assert np.all(np.diff(durs) >= -1e-9)
    assert np.all(np.diff(idl) <= 1e-9)


def test_enforce_pareto_monotonicity_clips_noise_without_touching_clean_data():
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Noisy: duration wiggles down at 0.3, idle wiggles up at 0.4.
    noisy_durations = [10.0, 12.0, 11.0, 13.0, 14.0]
    noisy_idles = [50.0, 40.0, 35.0, 38.0, 20.0]

    fixed_durations, fixed_idles = _enforce_pareto_monotonicity(
        percentiles, noisy_durations, noisy_idles,
    )

    assert fixed_durations == [10.0, 12.0, 12.0, 13.0, 14.0]
    assert fixed_idles == [50.0, 40.0, 35.0, 35.0, 20.0]

    # Already-clean, strictly monotonic data must pass through unchanged.
    clean_durations = [10.0, 11.0, 12.0, 13.0, 14.0]
    clean_idles = [50.0, 40.0, 30.0, 20.0, 10.0]
    same_durations, same_idles = _enforce_pareto_monotonicity(
        percentiles, clean_durations, clean_idles,
    )
    assert same_durations == clean_durations
    assert same_idles == clean_idles


def test_enforce_pareto_monotonicity_preserves_input_order():
    """Percentiles need not be pre-sorted; the fix-up must still return
    values aligned to the *original* input order, not the sorted one."""
    percentiles = [0.5, 0.1, 0.3]
    durations = [20.0, 10.0, 12.0]
    idles = [5.0, 50.0, 30.0]

    fixed_durations, fixed_idles = _enforce_pareto_monotonicity(
        percentiles, durations, idles,
    )

    # Already consistent with monotonic order (0.1->10, 0.3->12, 0.5->20),
    # so nothing should change.
    assert fixed_durations == durations
    assert fixed_idles == idles


def test_build_pareto_shift_updates_structure(tiny_task_file):
    full_progress = {
        1: {"status": "done", "actual_duration": 3.0},
        2: {"status": "done", "actual_duration": 4.5},
        3: {"status": "done", "actual_duration": 5.5},
        4: {"status": "done", "actual_duration": 6.5},
        5: {"status": "done", "actual_duration": 3.0},
    }

    points = build_pareto_shift_updates(
        full_progress=full_progress,
        task_file=tiny_task_file,
        base_percentile=0.7,
        n_updates=5,
    )

    # 1 baseline point + 5 updates * (drift, compensate) = 11 points
    assert len(points) == 1 + 5 * 2
    assert points[0]["kind"] == "baseline"
    assert points[0]["update"] == 0
    assert "effort" not in points[0], "effort is added later, by part5, off the baseline curve"

    kinds = [p["kind"] for p in points[1:]]
    assert kinds == ["drift", "compensate"] * 5

    # The very first drift shares the baseline's percentile (nothing has
    # been compensated yet); every later drift shares the percentile of
    # the *previous* compensate point — that's what makes the move
    # baseline/compensate → drift horizontal (same percentile).
    assert points[1]["percentile"] == points[0]["percentile"] == 0.7

    for update_num, (drift, compensate) in enumerate(
        zip(points[1::2], points[2::2]), start=1
    ):
        assert drift["update"] == update_num
        assert compensate["update"] == update_num
        assert drift["finish"] > 0
        assert compensate["finish"] > 0

    for prev_compensate, next_drift in zip(points[2::2], points[3::2]):
        assert next_drift["percentile"] == prev_compensate["percentile"]


def test_build_pareto_shift_updates_rebuilds_from_baseline_each_update(tiny_task_file):
    """Regression test for the plan-chaining bug.

    Each update must replan from the *original* baseline tasks — only the
    growing ``observed_progress`` dict and the scalar ``prev_percentile``
    may carry forward. Chaining the previous update's *replanned task
    list* forward would leak that update's (possibly aggressive)
    percentile into the next update's "planned_start" floor inside
    ``reconstruct_fact_schedule``, compounding into spurious extra drift
    unrelated to new information. This test recomputes the last update's
    compensate point completely independently, straight from an untouched
    baseline copy, and checks it matches exactly.
    """
    full_progress = {
        1: {"status": "done", "actual_duration": 3.6},
        2: {"status": "done", "actual_duration": 5.4},
        3: {"status": "done", "actual_duration": 6.6},
        4: {"status": "done", "actual_duration": 7.8},
        5: {"status": "done", "actual_duration": 3.6},
    }

    points = build_pareto_shift_updates(
        full_progress=full_progress,
        task_file=tiny_task_file,
        base_percentile=0.7,
        n_updates=5,
    )

    baseline = load_tasks_from_csv(tiny_task_file)
    build_schedule(baseline, percentile=0.7, seed=42)
    target = max(t.planned_end_time for t in baseline)

    independent = reschedule_with_fixed_project_deadline(
        tasks=copy.deepcopy(baseline),
        progress=full_progress,
        target_finish_time=target,
        current_time=None,
        base_percentile=0.7,
        prev_percentile=points[-2]["percentile"],  # previous update's drift percentile
        prior_mean=0.0,
        prior_std=0.3,
        obs_noise=0.1,
    )

    last_compensate = points[-1]
    assert last_compensate["kind"] == "compensate"
    assert last_compensate["finish"] == pytest.approx(independent["project_finish"], abs=1e-6)
    assert last_compensate["percentile"] == pytest.approx(independent["used_percentile"], abs=1e-6)


def test_build_pareto_shift_updates_uniform_bias_stabilizes_after_first_update():
    """Behavioral regression test: for a perfectly uniform bias, the bias
    posterior converges almost fully by the first update (few observations
    already pin it down tightly), so the compensate percentile should
    barely move between update 1 and update 2. Before the fix (chaining
    replanned tasks forward), this collapsed by ~20% between updates 1 and
    2 purely from the chaining artifact; after the fix it moves by only a
    few percent.
    """
    tasks = load_tasks_from_csv("data/tasks.csv")
    progress = {
        t.task_id: {"status": "done", "actual_duration": round(t.mean * 1.2, 4)}
        for t in tasks
    }

    points = build_pareto_shift_updates(
        full_progress=progress,
        task_file="data/tasks.csv",
        base_percentile=0.74,
        n_updates=5,
    )

    compensate_1 = points[2]["percentile"]  # update 1's compensate point
    compensate_2 = points[4]["percentile"]  # update 2's compensate point

    assert compensate_2 == pytest.approx(compensate_1, rel=0.15)


def test_compensate_points_carry_median_feasibility_diagnostic():
    """Every compensate point must report whether the deadline is still
    achievable at the bias-corrected median (percentile 0.5). For the
    uniform 1.2x overrun with no deadline buffer, the deadline is
    unreachable even at the median (see the dissertation discussion), so
    every compensate point should report median_feasible=False and a
    median_finish that overshoots the deadline.
    """
    tasks = load_tasks_from_csv("data/tasks.csv")
    progress = {
        t.task_id: {"status": "done", "actual_duration": round(t.mean * 1.2, 4)}
        for t in tasks
    }

    points = build_pareto_shift_updates(
        full_progress=progress,
        task_file="data/tasks.csv",
        base_percentile=0.74,
        n_updates=5,
    )

    compensate_points = [p for p in points if p["kind"] == "compensate"]
    assert len(compensate_points) == 5
    for pt in compensate_points:
        assert "median_finish" in pt
        assert "median_feasible" in pt
        assert pt["median_feasible"] is False
        # Median plan overshoots the deadline; once every task is "done"
        # (last update), percentile no longer affects anything and the two
        # values coincide exactly, so use >= rather than a strict >.
        assert pt["median_finish"] >= pt["finish"]


def test_deadline_buffer_scales_the_target_and_baseline_point():
    """`deadline_buffer` must inflate the target/baseline finish
    proportionally, and, given enough buffer, let a moderate overrun be
    absorbed without requiring an unachievable sub-median percentile."""
    tasks = load_tasks_from_csv("data/tasks.csv")
    progress = {
        t.task_id: {"status": "done", "actual_duration": round(t.mean * 1.05, 4)}
        for t in tasks
    }

    no_buffer = build_pareto_shift_updates(
        full_progress=progress, task_file="data/tasks.csv",
        base_percentile=0.74, n_updates=2,
    )
    with_buffer = build_pareto_shift_updates(
        full_progress=progress, task_file="data/tasks.csv",
        base_percentile=0.74, n_updates=2, deadline_buffer=0.15,
    )

    assert with_buffer[0]["finish"] == pytest.approx(no_buffer[0]["finish"] * 1.15)
    # With slack to spare, the buffered scenario's compensate points should
    # all be reported as achievable at the median.
    for pt in with_buffer:
        if pt["kind"] == "compensate":
            assert pt["median_feasible"] is True


def test_underrun_ceiling_stable_scenario_holds_percentile():
    """A severe-enough underrun should push the compensate percentile
    straight to max_percentile (0.99, the boundary — not an interior
    equilibrium) and keep it there for every subsequent update, since that
    boundary is a robust "as good as it gets" answer rather than a delicate
    match that new information can dislodge."""
    tasks = load_tasks_from_csv("data/tasks.csv")
    # Calibrated so the *estimated* posterior bias (which runs a few
    # percent above the raw multiplier, see the mean/median mismatch
    # discussion) comfortably clears the ceiling-pin threshold.
    progress = {
        t.task_id: {"status": "done", "actual_duration": round(t.mean * 0.57723, 5)}
        for t in tasks
    }

    points = build_pareto_shift_updates(
        full_progress=progress, task_file="data/tasks.csv",
        base_percentile=0.74, n_updates=5,
    )

    compensate_percentiles = [p["percentile"] for p in points if p["kind"] == "compensate"]
    for p in compensate_percentiles:
        assert p == pytest.approx(0.99, abs=0.02)


def test_overrun_buffered_stable_scenario_meets_deadline_throughout():
    """A moderate overrun combined with a genuine project buffer should
    keep the deadline met at every single update — the CCPM answer to a
    bias that would otherwise force an ever-shrinking, never-satisfied
    sub-median percentile."""
    tasks = load_tasks_from_csv("data/tasks.csv")
    progress = {
        t.task_id: {"status": "done", "actual_duration": round(t.mean * 1.2172, 5)}
        for t in tasks
    }

    points = build_pareto_shift_updates(
        full_progress=progress, task_file="data/tasks.csv",
        base_percentile=0.74, n_updates=5, deadline_buffer=0.15,
    )

    baseline_finish = points[0]["finish"]
    for pt in points:
        if pt["kind"] == "compensate":
            assert pt["finish"] <= baseline_finish + 1e-6


def test_batch_fraction_overrides_n_updates_batch_size(tiny_task_file):
    """`batch_fraction` decouples "how many updates" from "how large a
    share of tasks each update reveals" — n_updates=1 with
    batch_fraction=0.25 must produce exactly one (drift, compensate) pair
    built from ~25% of tasks, not from 100% (which is what n_updates=1
    would reveal under the default 1/n_updates sizing)."""
    tasks = load_tasks_from_csv(tiny_task_file)
    full_progress = {
        t.task_id: {"status": "done", "actual_duration": t.mean} for t in tasks
    }

    points = build_pareto_shift_updates(
        full_progress=full_progress, task_file=tiny_task_file,
        base_percentile=0.7, n_updates=1, batch_fraction=0.25,
    )

    # baseline + exactly one (drift, compensate) pair
    assert len(points) == 3
    assert [p["kind"] for p in points] == ["baseline", "drift", "compensate"]

    # With batch_fraction=0.25 on a 5-task fixture, only 1 task (round(5*0.25)=1)
    # should have been revealed — confirm via the observed-count implied by
    # the printed progress dict being partially, not fully, incorporated:
    # a full reveal (n_updates=1 default sizing) would have used all 5.
    default_sizing_points = build_pareto_shift_updates(
        full_progress=full_progress, task_file=tiny_task_file,
        base_percentile=0.7, n_updates=1,
    )
    assert points[1]["finish"] != pytest.approx(default_sizing_points[1]["finish"])


def test_single_recalculation_overrun_moves_effort_up_and_underrun_moves_it_down(
    tmp_path,
):
    """The two "one recalculation" showcase scenarios: a uniform overrun
    should force the compensate percentile down (raising idle time — the
    point moves *up* the idle axis), while a uniform underrun should let
    the compensate percentile rise to reclaim slack (lowering idle time —
    the point moves *down*)."""
    tasks = load_tasks_from_csv("data/tasks.csv")

    def uniform(factor):
        return {
            t.task_id: {"status": "done", "actual_duration": round(t.mean * factor, 4)}
            for t in tasks
        }

    overrun_points = build_pareto_shift_updates(
        full_progress=uniform(1.20), task_file="data/tasks.csv",
        base_percentile=0.74, n_updates=1, batch_fraction=0.25,
    )
    underrun_points = build_pareto_shift_updates(
        full_progress=uniform(0.70), task_file="data/tasks.csv",
        base_percentile=0.74, n_updates=1, batch_fraction=0.25,
    )

    for points in (overrun_points, underrun_points):
        assert len(points) == 3
        assert points[2]["kind"] == "compensate"

    # Overrun: percentile must drop below base_percentile (more idle risk).
    assert overrun_points[2]["percentile"] < 0.74
    # Underrun: percentile must rise above base_percentile (reclaimed slack).
    assert underrun_points[2]["percentile"] > 0.74


def test_plot_pareto_shift_trajectory_smoke(tmp_path):
    points = [
        {"kind": "baseline", "update": 0, "finish": 10.0, "effort": 0.0, "percentile": 0.7},
        {"kind": "drift", "update": 1, "finish": 13.0, "effort": 0.5, "percentile": 0.7},
        {"kind": "compensate", "update": 1, "finish": 10.5, "effort": 1.2, "percentile": 0.5},
        {"kind": "drift", "update": 2, "finish": 14.0, "effort": 1.0, "percentile": 0.5},
        {"kind": "compensate", "update": 2, "finish": 11.0, "effort": 2.0, "percentile": 0.4},
    ]
    out_file = tmp_path / "pareto_shift.png"

    plot_pareto_shift_trajectory(
        baseline_durations=[8.0, 10.0, 13.0],
        baseline_idles=[3.0, 1.5, 0.5],
        points=points,
        deadline=10.0,
        save_path=str(out_file),
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_part5_pareto_shift_trajectory_end_to_end(tiny_task_file, tmp_path):
    full_progress = {
        1: {"status": "done", "actual_duration": 2.5},
        2: {"status": "done", "actual_duration": 3.5},
    }
    out_file = tmp_path / "trajectory.png"

    points = part5_pareto_shift_trajectory(
        full_progress=full_progress,
        task_file=tiny_task_file,
        base_percentile=0.7,
        n_updates=5,
        baseline_percentiles=[0.3, 0.6, 0.9],
        baseline_n_iter=100,
        save_path=str(out_file),
    )

    assert len(points) == 1 + 5 * 2
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_part5_effort_tracks_percentile_on_baseline_curve(tiny_task_file, tmp_path):
    """Effort is added by part5 as a function of percentile alone: two
    points sharing a percentile must land at the same height, and every
    point's effort must come straight from the interpolated baseline curve."""
    full_progress = {
        1: {"status": "done", "actual_duration": 4.0},
        2: {"status": "done", "actual_duration": 5.0},
        3: {"status": "done", "actual_duration": 6.0},
    }
    out_file = tmp_path / "trajectory.png"
    baseline_percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]

    points = part5_pareto_shift_trajectory(
        full_progress=full_progress,
        task_file=tiny_task_file,
        base_percentile=0.7,
        n_updates=5,
        baseline_percentiles=baseline_percentiles,
        baseline_n_iter=150,
        save_path=str(out_file),
    )

    for pt in points:
        assert "effort" in pt
        assert pt["effort"] >= 0

    baseline_point, first_drift = points[0], points[1]
    assert first_drift["percentile"] == baseline_point["percentile"]
    assert first_drift["effort"] == pytest.approx(baseline_point["effort"])
