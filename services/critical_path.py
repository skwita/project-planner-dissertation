from collections import defaultdict, deque

def find_critical_path(tasks):
    task_dict = {t.task_id: t for t in tasks}

    # Строим граф зависимостей
    graph = defaultdict(list)
    indegree = defaultdict(int)
    for task in tasks:
        for dep in task.dependencies:
            graph[dep].append(task.task_id)
            indegree[task.task_id] += 1
        indegree.setdefault(task.task_id, 0)

    # Топологическая сортировка
    queue = deque([tid for tid, deg in indegree.items() if deg == 0])
    topo_order = []
    while queue:
        tid = queue.popleft()
        topo_order.append(tid)
        for next_tid in graph[tid]:
            indegree[next_tid] -= 1
            if indegree[next_tid] == 0:
                queue.append(next_tid)

    # Рассчитываем ранние старты и окончания
    earliest_start = {t.task_id: 0 for t in tasks}
    earliest_finish = {}
    for tid in topo_order:
        t = task_dict[tid]
        start = max([earliest_finish[d] for d in t.dependencies] or [0])
        earliest_start[tid] = start
        earliest_finish[tid] = start + t.mean

    # Находим конец проекта
    project_end = max(earliest_finish.values())

    # Ищем критический путь (обратным ходом)
    critical_path = []
    # Задачи, заканчивающиеся в конце проекта
    current = [tid for tid, finish in earliest_finish.items() if finish == project_end][0]

    while True:
        critical_path.append(task_dict[current])
        preds = [
            d for d in task_dict[current].dependencies
            if abs(earliest_finish[d] - earliest_start[current]) < 1e-6
        ]
        if preds:
            current = preds[0]
        else:
            break

    critical_path.reverse()
    return critical_path

def critical_chain_path(tasks):
    """
    Построить "критическую цепь" (critical chain) — т.е. путь, учитывающий
    и зависимости, и ресурсную последовательность (по role/planned_start_time).

    Возвращает список Task в порядке выполнения вдоль найденного пути.
    """

    # Словарь по id
    task_by_id = {t.task_id: t for t in tasks}

    # Вычисляем длительность каждой задачи:
    # используем planned_end_time - planned_start_time, если оба заданы,
    # иначе fallback на mean.
    duration = {}
    for t in tasks:
        if t.planned_start_time is not None and t.planned_end_time is not None:
            duration[t.task_id] = float(t.planned_end_time - t.planned_start_time)
        else:
            duration[t.task_id] = float(t.mean)

    # Строим граф: добавляем рёбра для зависимостей
    graph = defaultdict(list)
    indeg = defaultdict(int)
    for t in tasks:
        indeg.setdefault(t.task_id, 0)
    for t in tasks:
        for d in t.dependencies or []:
            if d not in task_by_id:
                # игнорируем несуществующие зависимости
                continue
            graph[d].append(t.task_id)
            indeg[t.task_id] += 1

    # Добавляем рёбра ресурсной последовательности (по role)
    # Для каждой роли сортируем её задачи по planned_start_time (если есть),
    # иначе по planned_end_time, иначе по task_id.
    role_groups = defaultdict(list)
    for t in tasks:
        role_groups[t.role].append(t)

    for role, tlist in role_groups.items():
        # ключ сортировки: None -> large so tasks without times go last by id
        def sort_key(x):
            if x.planned_start_time is not None:
                return (0, float(x.planned_start_time))
            if x.planned_end_time is not None:
                return (1, float(x.planned_end_time))
            return (2, int(x.task_id))
        ordered = sorted(tlist, key=sort_key)
        for a, b in zip(ordered, ordered[1:]):
            # добавляем ребро a -> b (a выполняется раньше, b позже)
            graph[a.task_id].append(b.task_id)
            indeg[b.task_id] += 1

    # Топологическая сортировка — если получился цикл, попробуем ломать циклы
    # ориентируя ресурcные рёбра по стартам (мы их уже ориентировали),
    # но цикл всё ещё возможен при конфликтующих данных; в таком случае
    # удалим рёбра между задачами одной роли, ориентируясь по start_time (последовательно).
    def topo_sort(graph, indeg):
        q = deque([tid for tid, d in indeg.items() if d == 0])
        topo = []
        indeg_copy = dict(indeg)
        while q:
            u = q.popleft()
            topo.append(u)
            for v in graph.get(u, []):
                indeg_copy[v] -= 1
                if indeg_copy[v] == 0:
                    q.append(v)
        return topo, indeg_copy

    topo, indeg_left = topo_sort(graph, indeg)
    if len(topo) != len(tasks):
        # обнаружен цикл — попробуем удалить минимальное число ресурсных рёбер,
        # ориентируясь на planned_start_time (пожертвуем тем ребром, которое нарушает порядок)
        # Простая эвристика: для каждой роли убедимся, что последовательность образует прямые рёбра
        # и если где-то возникает цикл, убираем ресурсное ребро.
        # (это аккуратная эвристика; для строгой корректности можно применять алгоритмы поиска минимального обратного набора рёбер)
        resource_edges = set()
        for role, tlist in role_groups.items():
            ordered = sorted(tlist, key=lambda x: (x.planned_start_time if x.planned_start_time is not None else float('inf'),
                                                   x.planned_end_time if x.planned_end_time is not None else float('inf'),
                                                   x.task_id))
            for a, b in zip(ordered, ordered[1:]):
                resource_edges.add((a.task_id, b.task_id))

        # Попробуем поочередно удалить проблемные ресурсные рёбра до тех пор, пока топо не станет полным
        graph2 = {k: list(v) for k, v in graph.items()}
        indeg2 = dict(indeg)
        for (a, b) in list(resource_edges):
            if len(topo) == len(tasks):
                break
            # удаляем ребро a->b, если оно есть
            if b in graph2.get(a, []):
                graph2[a].remove(b)
                indeg2[b] -= 1
                topo, indeg_left = topo_sort(graph2, indeg2)
        # используем финальный граф
        graph = graph2
        indeg = indeg2
        topo, indeg_left = topo_sort(graph, indeg)
        if len(topo) != len(tasks):
            raise RuntimeError("Не удалось получить ацикличный граф для построения критического пути (остались циклы). Проверьте зависимости и плановые времена.")

    # Теперь считаем максимальную суммарную длительность пути до каждой вершины (longest path)
    # dist[tid] = максимальная суммарная длительность пути, заканчивающегося в tid
    dist = {t.task_id: float('-inf') for t in tasks}
    prev = {t.task_id: None for t in tasks}

    # Инициализируем стартовые вершины (у которых нет входящих рёбер в текущем графе)
    for tid in topo:
        if indeg.get(tid, 0) == 0:
            dist[tid] = duration[tid]

    # Проходим в топологическом порядке
    for u in topo:
        if dist[u] == float('-inf'):
            # вершина недостижима от начальных (это может случиться, если были удалены рёбра)
            dist[u] = duration[u]
        for v in graph.get(u, []):
            cand = dist[u] + duration[v]
            if cand > dist[v]:
                dist[v] = cand
                prev[v] = u

    # Находим вершину с максимальным dist (конец критической цепи)
    end_tid = max(dist.keys(), key=lambda k: dist[k])
    # Восстанавливаем путь
    path_ids = []
    cur = end_tid
    while cur is not None:
        path_ids.append(cur)
        cur = prev[cur]
    path_ids.reverse()
    return [task_by_id[i] for i in path_ids]