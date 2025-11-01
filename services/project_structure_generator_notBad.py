import random
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def generate_realistic_tasks(num_tasks=50, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    roles = ["аналитик", "разработчик", "тестировщик"]

    def analyst_prob(x): return np.exp(-((x - 0.1) ** 2) / 0.05)
    def developer_prob(x): return np.exp(-((x - 0.55) ** 2) / 0.05)
    def tester_prob(x): return np.exp(-((x - 0.9) ** 2) / 0.05)

    role_funcs = {"аналитик": analyst_prob, "разработчик": developer_prob, "тестировщик": tester_prob}
    time_ranges = {"аналитик": (8, 40), "разработчик": (16, 80), "тестировщик": (8, 40)}

    tasks = []
    all_ids = list(range(1, num_tasks + 1))

    # Разделим проект на фазы
    n_phases = 5
    phase_boundaries = np.linspace(0, num_tasks, n_phases + 1, dtype=int)
    phases = [list(range(phase_boundaries[i], phase_boundaries[i + 1])) for i in range(n_phases)]
    phases = [p for p in phases if p]

    for i in range(num_tasks):
        progress = i / (num_tasks - 1)

        weights = np.array([role_funcs[r](progress) for r in roles])
        weights /= weights.sum()
        role = random.choices(roles, weights=weights)[0]

        mean_time = round(random.uniform(*time_ranges[role]), 1)
        std_dev = round(mean_time * random.uniform(0.05, 0.25), 1)

        if i == 0:
            predecessors = ""
        else:
            possible_predecessors = []

            back_window = min(7, i)
            possible_predecessors.extend(all_ids[i - back_window:i])

            if random.random() < 0.25:
                prev_phase_idx = next((p for p, phase in enumerate(phases) if i in phase), 0)
                if prev_phase_idx > 0:
                    possible_predecessors.extend(phases[prev_phase_idx - 1])

            if i > 2 and random.random() < 0.1:
                jump_back = random.randint(2, min(10, i))
                possible_predecessors.append(i + 1 - jump_back)

            if i > 3 and random.random() < 0.05:
                possible_predecessors.append(random.choice(all_ids[:i - 2]))

            possible_predecessors = list(set(p for p in possible_predecessors if p < i + 1))

            if possible_predecessors:
                n_deps = random.randint(1, min(3, len(possible_predecessors)))
                predecessors = ", ".join(map(str, random.sample(possible_predecessors, n_deps)))
            else:
                predecessors = str(i)

        tasks.append({
            "task_id": i + 1,
            "role": role.lower(),
            "dependencies": predecessors,
            "mean": mean_time,
            "stddev": std_dev
        })

    # ====== Удаляем транзитивные зависимости ======
    dep_map = defaultdict(list)
    for t in tasks:
        if t["dependencies"]:
            for d in t["dependencies"].split(", "):
                dep_map[t["task_id"]].append(int(d))

    def get_all_predecessors(task_id):
        """Собираем все достижимые зависимости (по цепочке)."""
        visited = set()
        queue = deque(dep_map[task_id])
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                for nxt in dep_map.get(cur, []):
                    queue.append(nxt)
        return visited

    for t in tasks:
        if not t["dependencies"]:
            continue
        deps = [int(x) for x in t["dependencies"].split(", ")]
        reachable = set()
        for d in deps:
            reachable |= get_all_predecessors(d)
        # оставляем только те зависимости, которые не достижимы через другие
        filtered = [d for d in deps if d not in reachable]
        t["dependencies"] = ", ".join(map(str, filtered))

    # =================================================

    df = pd.DataFrame(tasks)
    df.to_csv("generated_tasks.csv", index=False, encoding="utf-8-sig")
    return df

# === Визуализация графа ===
def plot_dependency_graph(df):
    G = nx.DiGraph()

    # Добавляем узлы и зависимости
    for _, row in df.iterrows():
        G.add_node(row["task_id"], role=row["role"])
        if isinstance(row["dependencies"], str) and row["dependencies"].strip():
            for dep in map(str.strip, row["dependencies"].split(",")):
                if dep:
                    G.add_edge(int(dep), row["task_id"])

    # Определяем "уровень" каждой задачи — глубину зависимостей
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if preds:
            levels[node] = max(levels[p] for p in preds) + 1
        else:
            levels[node] = 0

    # Группировка узлов по уровням
    level_nodes = defaultdict(list)
    for node, lvl in levels.items():
        level_nodes[lvl].append(node)

    # Задаём координаты (x = уровень, y = равномерно внутри уровня)
    pos = {}
    for lvl, nodes in sorted(level_nodes.items()):
        y_positions = list(range(len(nodes)))
        y_center = (len(nodes) - 1) / 2
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (lvl, y_center - i)

    # Цвета по ролям
    role_colors = {
        "аналитик": "#6baed6",    # голубой
        "разработчик": "#74c476",  # зелёный
        "тестировщик": "#fd8d3c"      # оранжевый
    }
    node_colors = [role_colors[G.nodes[n]["role"]] for n in G.nodes]

    # Отрисовка
    plt.figure(figsize=(16, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=900,
        font_size=9,
        arrowsize=12,
        edge_color="gray",
        font_color="black",
        connectionstyle="arc3,rad=0.05"  # лёгкая кривизна
    )

    # Легенда
    handles = [
        Line2D([0], [0], marker='o', color='w',
               label=r.capitalize(),
               markerfacecolor=c, markersize=10,
               markeredgecolor='black')
        for r, c in role_colors.items()
    ]
    plt.legend(handles=handles, title="Roles", loc="upper left")
    plt.title("Project Dependency Graph (Left → Right, Layered Layout)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = generate_realistic_tasks(num_tasks=50)
    # df = generate_realistic_tasks(num_tasks=50, random_seed=42)
    print(df.head(10))
    print(f"\nGenerated {len(df)} tasks. Saved to generated_tasks.csv.")
    plot_dependency_graph(df)
