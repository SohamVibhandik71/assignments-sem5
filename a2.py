
from collections import deque
import time

graph = {
    'A': ['B', 'C'],
    'B': ['H'],
    'C': ['D'],
    'D': ['E'],
    'E': ['H'],
    'H': []
}


def bfs(start, goal):
    q = deque([(start, [start])])
    visited = set()
    nodes = 0

    while q:
        node, path = q.popleft()

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for nxt in graph[node]:
            if nxt not in visited:
                q.append((nxt, path + [nxt]))

    return None, nodes


def dfs(start, goal):
    stack = [(start, [start])]
    visited = set()
    nodes = 0

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for nxt in graph[node]:
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))

    return None, nodes


for name, search in [('BFS', bfs), ('DFS', dfs)]:
    start = time.perf_counter()
    path, nodes = search('A', 'H')
    elapsed = time.perf_counter() - start

    print(f"\n{name} Results")
    print("Path:", " -> ".join(path) if path else "No path found")
    print("Path Length:", len(path) - 1 if path else 0)
    print("Nodes Explored:", nodes)
    print("Execution Time:", f"{elapsed:.8f} seconds")

print("\nBFS finds the shortest path in an unweighted graph.")
print("DFS may find a longer path depending on traversal order.")
