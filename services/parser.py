"""CSV parser for task definitions."""

import pandas as pd
from models.task import Task


def load_tasks_from_csv(path: str) -> list[Task]:
    """
    Load tasks from a CSV file with columns:
        task_id, role, dependencies, mean, stddev

    The ``dependencies`` column is a comma-separated list of task IDs,
    or empty for tasks with no predecessors.

    Args:
        path: Path to the CSV file.

    Returns:
        List of Task objects in file order.
    """
    df = pd.read_csv(path)
    tasks = []
    for _, row in df.iterrows():
        raw_deps = row["dependencies"]
        dependencies = (
            list(map(int, str(raw_deps).split(",")))
            if pd.notna(raw_deps) and str(raw_deps).strip()
            else []
        )
        tasks.append(Task(
            task_id=row["task_id"],
            role=row["role"],
            dependencies=dependencies,
            mean=row["mean"],
            stddev=row["stddev"],
        ))
    return tasks
