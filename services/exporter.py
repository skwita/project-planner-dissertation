"""Excel export helpers."""

import pandas as pd

from models.task import Task


def export_schedule_to_excel(
    tasks: list[Task],
    filename: str,
    project_duration: float,
    idle_time: dict[str, float] | None = None,
) -> None:
    """
    Write the schedule to an Excel workbook with two sheets:
      - 'Schedule': one row per task plus a totals row.
      - 'Idle by role': per-role idle time (omitted when idle_time is None).

    Args:
        tasks:            Scheduled tasks with all time fields set.
        filename:         Output path (e.g. ``"output/schedule.xlsx"``).
        project_duration: Real project duration to write in the totals row.
        idle_time:        Optional dict mapping role → idle days.
    """
    schedule_rows = [
        {
            "ID": t.task_id,
            "Role": t.role,
            "Predecessors": ", ".join(map(str, t.dependencies)) if t.dependencies else "",
            "Mean duration": t.mean,
            "Std dev": t.stddev,
            "Planned start": round(t.planned_start_time, 2),
            "Planned duration": round(t.planned_duration, 2),
            "Planned end": round(t.planned_end_time, 2),
            "Real start": round(t.real_start_time, 2),
            "Real duration": round(t.real_duration, 2),
            "Real end": round(t.real_end_time, 2),
        }
        for t in tasks
    ]
    df_schedule = pd.DataFrame(schedule_rows)
    # Append totals row
    totals = {col: "" for col in df_schedule.columns}
    totals["ID"] = "TOTAL"
    totals["Real end"] = round(project_duration, 2)
    df_schedule = pd.concat(
        [df_schedule, pd.DataFrame([totals])], ignore_index=True
    )

    idle_rows = (
        [{"Role": role, "Idle (days)": round(days, 2)} for role, days in idle_time.items()]
        if idle_time else []
    )
    df_idle = pd.DataFrame(idle_rows)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_schedule.to_excel(writer, index=False, sheet_name="Schedule")
        if not df_idle.empty:
            df_idle.to_excel(writer, index=False, sheet_name="Idle by role")


def export_percentile_analysis_to_excel(results: list[dict], output_path: str) -> pd.DataFrame:
    """
    Write percentile analysis results to Excel and return the DataFrame.

    Args:
        results:     List of result dicts (one per percentile).
        output_path: Output file path.

    Returns:
        The resulting DataFrame.
    """
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    return df
