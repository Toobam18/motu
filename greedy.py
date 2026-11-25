"""
Greedy Best-First Search for MOTU
=================================

Lecture definition (Chapter 4):
    "Expand the node that appears to be closest to the goal."
    Evaluation function  f(n) = h(n)   (g(n) is ignored)
    Fringe = priority queue ordered by h(n), lowest first.

Properties from the lecture:
    Complete?  Graph search with an explored set: yes on our finite graphs.
               (Tree search can get stuck in loops.)
    Optimal?   NO. It ignores the cost already paid, so it can take an
               expensive edge just because it points toward the goal.
    Time/space: often very fast with a good heuristic; worst case O(b^m).

Heuristic: same straight-line distance * h_scale as A* (script.py).

The returned "cost" is the REAL travel time of the path Greedy chose, so it
can be compared with the optimal cost found by UCS / A*.
"""

import heapq
import time

from script import euclidean_heuristic
from USC import successor_fn


def greedy_best_first_search(graph, start, goal):
    """Graph-search version of Greedy Best-First Search.

    Returns: found, path, cost, nodes_expanded, time_ms, explored
    """
    t0 = time.perf_counter()

    if start not in graph:
        raise ValueError(f"Unknown start location: {start}")
    if goal not in graph:
        raise ValueError(f"Unknown goal location: {goal}")

    def h(state):
        return euclidean_heuristic(graph, state, goal)

    # FRINGE ordered by h(n) only. The state name breaks ties so the run is
    # deterministic. g is carried along just to report the real path cost.
    fringe = [(h(start), start)]
    path_cost = {start: 0}
    parent = {start: None}
    explored = set()

    while fringe:
        # Remove-Front: the node that LOOKS closest to the goal.
        _, state = heapq.heappop(fringe)

        if state in explored:
            continue
        explored.add(state)

        if state == goal:
            solution = []
            node = state
            while node is not None:
                solution.append(node)
                node = parent[node]
            return {
                "found": True,
                "path": solution[::-1],
                "cost": path_cost[goal],
                "nodes_expanded": len(explored),
                "explored": explored,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }

        # EXPAND: add each new successor once. Greedy does not care about
        # cheaper paths (it never looks at g), so a state already on the
        # fringe keeps the parent it was first reached from.
        for next_state, step_cost in successor_fn(graph, state):
            if next_state not in parent:
                parent[next_state] = state
                path_cost[next_state] = path_cost[state] + step_cost
                heapq.heappush(fringe, (h(next_state), next_state))

    return {
        "found": False,
        "path": [],
        "cost": float("inf"),
        "nodes_expanded": len(explored),
        "explored": explored,
        "time_ms": (time.perf_counter() - t0) * 1000,
    }


# Short name for app.py / experiments
greedy = greedy_best_first_search


if __name__ == "__main__":
    from script import G
    print("Locations:", ", ".join(sorted(G.nodes)))
    start = input("Start: ").strip().upper() or "SHA"
    goal = input("Goal:  ").strip().upper() or "BUS"
    r = greedy_best_first_search(G, start, goal)
    if r["found"]:
        print("Route:", " -> ".join(r["path"]), f"| cost {r['cost']} s")
    else:
        print("Failure: no route exists")
    print("Nodes expanded:", r["nodes_expanded"], f"| {r['time_ms']:.3f} ms")
