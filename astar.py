"""
A* Search for MOTU
==================

Lecture definition (Chapter 4):
    "Avoid expanding paths that are already expensive."
    Evaluation function  f(n) = g(n) + h(n)
        g(n) = cost so far to reach n
        h(n) = estimated cost from n to the goal
        f(n) = estimated total cost of the cheapest path through n
    Fringe = priority queue ordered by f(n), lowest first.

Properties from the lecture:
    Complete?  Yes (finite graph, step costs > 0).
    Optimal?   Yes. Graph search with an explored set needs h to be
               CONSISTENT: h(n) <= c(n, n') + h(n'). Our straight-line
               heuristic is consistent (see euclidean_heuristic in script.py),
               so the first time the goal is removed from the fringe its path
               is the cheapest one.
    Informed:  uses h(n), so it usually expands far fewer nodes than UCS.

Heuristic:
    h(n) = straight-line distance from n to the goal * h_scale (seconds per
    map unit). Every edge costs at least its length * h_scale, so h never
    overestimates (admissible). With h = 0, A* behaves exactly like UCS.

Same problem formulation and return format as USC.py, so the front end and
experiments can compare the three algorithms side by side.
"""

import heapq
import time

from script import euclidean_heuristic
from USC import successor_fn   # same successor function as UCS


def a_star_search(graph, start, goal, h_weight=1.0):
    """Graph-search version of A*.

    h_weight scales the heuristic for the "heuristic quality" experiments:
    1.0 = full straight-line heuristic, 0.5 = weaker, 0 = UCS.

    Returns: found, path, cost, nodes_expanded, time_ms, explored
    """
    t0 = time.perf_counter()

    if start not in graph:
        raise ValueError(f"Unknown start location: {start}")
    if goal not in graph:
        raise ValueError(f"Unknown goal location: {goal}")

    def h(state):
        return euclidean_heuristic(graph, state, goal, scale=h_weight)

    # FRINGE: entries are (f(n), g(n), state). heapq pops the lowest f(n);
    # ties are broken by the lower g(n).
    fringe = [(h(start), 0, start)]
    path_cost = {start: 0}          # g(n): cheapest known cost to each state
    parent = {start: None}          # parent pointers to rebuild the path
    explored = set()                # closed list: states already expanded

    while fringe:
        # Remove-Front: the node with the LOWEST f(n) = g(n) + h(n).
        # Using f instead of g is the only difference from UCS.
        f, g, state = heapq.heappop(fringe)

        if state in explored:       # stale, more expensive copy: skip it
            continue
        explored.add(state)

        # GOAL TEST when the node is removed (expanded), not when it's added.
        # With a consistent h, this first goal popped is optimal.
        if state == goal:
            solution = []
            while state is not None:
                solution.append(state)
                state = parent[state]
            return {
                "found": True,
                "path": solution[::-1],
                "cost": g,
                "nodes_expanded": len(explored),
                "explored": explored,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }

        # EXPAND: g(n') = g(n) + step cost,  f(n') = g(n') + h(n')
        for next_state, step_cost in successor_fn(graph, state):
            new_g = g + step_cost
            if next_state not in path_cost or new_g < path_cost[next_state]:
                path_cost[next_state] = new_g
                parent[next_state] = state
                heapq.heappush(fringe, (new_g + h(next_state), new_g, next_state))

    return {
        "found": False,
        "path": [],
        "cost": float("inf"),
        "nodes_expanded": len(explored),
        "explored": explored,
        "time_ms": (time.perf_counter() - t0) * 1000,
    }


# Short name for app.py / experiments
a_star = a_star_search


if __name__ == "__main__":
    from script import G
    print("Locations:", ", ".join(sorted(G.nodes)))
    start = input("Start: ").strip().upper() or "SHA"
    goal = input("Goal:  ").strip().upper() or "BUS"
    r = a_star_search(G, start, goal)
    if r["found"]:
        print("Route:", " -> ".join(r["path"]), f"| cost {r['cost']} s")
    else:
        print("Failure: no route exists")
    print("Nodes expanded:", r["nodes_expanded"], f"| {r['time_ms']:.3f} ms")
