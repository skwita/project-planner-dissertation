import pandas as pd

def export_schedule_to_excel(tasks, filename, project_duration, idle_time=None):
    # === Первый лист: План проекта ===
    data = []

    for task in tasks:
        data.append({
            "ID": task.task_id,
            "Роль": task.role,
            "Предшественники": ", ".join(map(str, task.dependencies)) if task.dependencies else "",
            "Ожидаемое ср.время": task.mean,
            "Абсолютное отклонение": task.stddev,
            "Запланированное начало":round(task.planned_start_time, 2),
            "Запланированная длительность":round(task.planned_duration, 2),
            "Запланированный конец":round(task.planned_end_time, 2),
            "Фактическое начало":round(task.real_start_time, 2),
            "Фактическая длительность":round(task.real_duration, 2),
            "Фактический конец":round(task.real_end_time, 2)
        })

    df_schedule = pd.DataFrame(data)
    df_schedule.loc[len(df_schedule.index)] = {
        "ID": "ИТОГО",
        "Реальное окончание": round(project_duration, 2)
    }

    # === Второй лист: Простой по ролям ===
    if idle_time is not None:
        idle_data = [{"Роль": role, "Простой (дней)": round(duration, 2)} for role, duration in idle_time.items()]
        df_idle = pd.DataFrame(idle_data)
    else:
        df_idle = pd.DataFrame()

    # === Запись в Excel с двумя листами ===
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_schedule.to_excel(writer, index=False, sheet_name="План проекта")
        if not df_idle.empty:
            df_idle.to_excel(writer, index=False, sheet_name="Простой по ролям")

def export_percentile_analysis_to_excel(results, output_path):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    return df

