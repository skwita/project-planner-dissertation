"""Tests for the Pareto-shift trajectory: data collection and plotting."""

import pytest

from main import (
    build_pareto_shift_updates,
    compute_pareto_idle_duration_curve,
    part5_pareto_shift_trajectory,
)
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
