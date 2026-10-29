
import heapq
import time

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 2, 'E': 5},
    'C': {'F': 3},
    'D': {'G': 4},
    'E': {'G': 2},
    'F': {'G': 2},
    'G': {}
}

h = {'A': 7, 'B': 6, 'C': 4, 'D': 4, 'E': 2, 'F': 2, 'G': 0}


def greedy(start, goal):
    pq = [(h[start], start, [start])]
    visited = set()
    nodes = 0

    while pq:
        _, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for nxt in graph[node]:
            if nxt not in visited:
                heapq.heappush(pq, (h[nxt], nxt, path + [nxt]))

    return None, nodes


def astar(start, goal):
    pq = [(h[start], 0, start, [start])]
    cost = {start: 0}
    nodes = 0

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if g > cost[node]:
            continue

        nodes += 1

        if node == goal:
            return path, g, nodes

        for nxt, c in graph[node].items():
            new_g = g + c

            if new_g < cost.get(nxt, float('inf')):
                cost[nxt] = new_g
                heapq.heappush(pq, (new_g + h[nxt], new_g, nxt, path + [nxt]))

    return None, float('inf'), nodes


start, goal = 'A', 'G'

for name, search in [('Greedy', greedy), ('A*', astar)]:
    t = time.perf_counter()
    result = search(start, goal)
    elapsed = time.perf_counter() - t

    path = result[0]
    nodes = result[-1]

    path_cost = sum(graph[path[i]][path[i+1]]
                    for i in range(len(path)-1)) if path else float('inf')

    print(f"\n{name} Search")
    print("Path:", " -> ".join(path) if path else "No path")
    print("Cost:", path_cost)
    print("Nodes explored:", nodes)
    print("Time:", f"{elapsed:.8f} seconds")
