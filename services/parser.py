import pandas as pd
from models.task import Task

def load_tasks_from_csv(path):
    df = pd.read_csv(path)
    tasks = []
    for _, row in df.iterrows():
        deps = row['dependencies']
        dependencies = deps.split(',') if pd.notna(deps) else []
        dependencies = list(map(int, dependencies))

        task = Task(
            task_id=row['task_id'],
            role=row['role'],
            dependencies=dependencies,
            mean=row['mean'],
            stddev=row['stddev']
        )
        tasks.append(task)
    return tasks

