"""
Uniform-Cost Search (UCS) for MOTU
==================================

Lecture definition (Chapter 3):
    "Expand least-cost unexpanded node."
    Implementation: fringe = queue ordered by path cost, lowest first.

Properties from the lecture:
    Complete?  Yes, if every step cost > 0 (all our travel times are > 0).
    Optimal?   Yes, because nodes are expanded in increasing order of g(n).
    Uninformed: UCS has no heuristic h(n). It only uses g(n), the path cost
                from the initial state to node n.

Problem formulation used here (from our proposal):
    State        = a campus location (a node in the graph)
    Initial state = start location chosen by the user
    Actions      = walking along an edge to a connected location
    Step cost    = the edge "weight" (travel time in seconds)
    Goal test    = are we at the goal location?
    Path cost    = g(n) = sum of step costs from start to n

The graph itself (environment) comes from script.py. networkx only STORES
the graph; the search below is written by us, as the project spec requires.
"""

import heapq   # priority queue: always removes the item with the lowest value
import time

import networkx as nx            # only used to DRAW the graph here
import matplotlib.pyplot as plt  # draws the map window

from script import G, POSITIONS  # the campus graph + where to draw each building


def successor_fn(graph, state):
    """Successor function: the actions available from a state.

    Returns (next_state, step_cost) for every connected location.
    Edges with an unknown time (weight = None) are skipped, since we
    can't cost a path through them.
    """
    for next_state in graph.neighbors(state):
        step_cost = graph[state][next_state]["weight"]
        if step_cost is not None:
            yield next_state, step_cost


def uniform_cost_search(graph, start, goal):
    """Graph-search version of UCS: Tree-Search + an explored set.

    Returns a dict with the metrics from our evaluation plan:
        found, path, cost, nodes_expanded, time_ms
    """
    t0 = time.perf_counter()

    # FIX 4: reject unknown locations up front (an unknown goal used to
    # search the whole map and quietly return found=False)
    if start not in graph:
        raise ValueError(f"Unknown start location: {start}")
    if goal not in graph:
        raise ValueError(f"Unknown goal location: {goal}")

    # FRINGE (a.k.a. frontier / open list): nodes waiting to be expanded.
    # Each entry is (g(n), state). heapq keeps it ordered by path cost,
    # lowest first, exactly like the lecture's UCS implementation.
    fringe = [(0, start)]

    # g(n): the cheapest path cost found so far to each state.
    path_cost = {start: 0}

    # Parent pointers. In the lecture, each node stores its Parent-Node;
    # we follow these backwards at the end to build the solution path.
    parent = {start: None}

    # EXPLORED set (a.k.a. closed list): states already expanded.
    # This turns Tree-Search into Graph-Search so we never expand the same
    # building twice (our campus graph has loops, e.g. ENG-LIB-UA-ENG).
    explored = set()

    # Main loop, same structure as Tree-Search in the lecture:
    #   if fringe is empty -> failure
    #   node <- Remove-Front(fringe)
    #   if Goal-Test(node) -> return solution
    #   fringe <- InsertAll(Expand(node), fringe)
    while fringe:

        # Remove-Front: take the node with the LOWEST g(n).
        # This single line is what makes it Uniform-Cost Search.
        g, state = heapq.heappop(fringe)

        # A state can be in the fringe more than once if we found a
        # cheaper path later. Skip the old, more expensive copies.
        if state in explored:
            continue
        explored.add(state)

        # GOAL TEST, done when the node is REMOVED (expanded), not when it
        # is added. Because nodes come out in increasing order of g(n), the
        # first time the goal comes out, no cheaper path can exist.
        # That is why UCS is optimal.
        if state == goal:
            solution = []
            while state is not None:          # follow parent pointers back
                solution.append(state)
                state = parent[state]
            return {
                "found": True,
                "path": solution[::-1],        # reverse: start -> goal
                "cost": g,
                "nodes_expanded": len(explored),
                "explored": explored,          # used by the visualization
                "time_ms": (time.perf_counter() - t0) * 1000,
            }

        # EXPAND the node: generate its successors.
        # Lecture: Path-Cost[s] = Path-Cost[node] + Step-Cost(node, action, s)
        for next_state, step_cost in successor_fn(graph, state):
            new_g = g + step_cost

            # Only add/update the successor if this is the cheapest path
            # to it we've seen so far.
            if next_state not in path_cost or new_g < path_cost[next_state]:
                path_cost[next_state] = new_g
                parent[next_state] = state
                heapq.heappush(fringe, (new_g, next_state))

    # Fringe is empty and the goal was never reached -> failure (no route).
    return {
        "found": False,
        "path": [],
        "cost": float("inf"),
        "nodes_expanded": len(explored),
        "explored": explored,
        "time_ms": (time.perf_counter() - t0) * 1000,
    }


def show_path(graph, start, goal, result, save_path=None):
    """Visualization: draw the campus map and highlight the UCS route.

    - Grey edges/nodes:  the whole campus graph (the state space)
    - Orange nodes:      locations UCS expanded but didn't use
    - Red edges/nodes:   the solution path UCS returned
    - Green / blue:      start and goal
    Edge labels show the step cost (travel time in seconds).
    """
    pos = graph.graph.get("positions", POSITIONS)   # FIX 5: was POSITIONS (campus only)
    path = result["path"]
    path_edges = list(zip(path, path[1:]))   # consecutive pairs = edges used

    fig, ax = plt.subplots(figsize=(8, 9))

    # 1. Whole map in grey
    nx.draw_networkx_edges(graph, pos, edge_color="lightgrey", width=2, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color="lightgrey",
                           edgecolors="black", node_size=1400, ax=ax)

    # 2. Locations UCS expanded (explored set) but that aren't on the path
    explored_only = [n for n in result.get("explored", []) if n not in path]
    nx.draw_networkx_nodes(graph, pos, nodelist=explored_only,
                           node_color="orange", edgecolors="black",
                           node_size=1400, ax=ax)

    # 3. The solution path in red
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                           edge_color="red", width=5, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color="salmon",
                           edgecolors="black", node_size=1400, ax=ax)

    # 4. Start (green) and goal (blue) on top
    nx.draw_networkx_nodes(graph, pos, nodelist=[start], node_color="limegreen",
                           edgecolors="black", node_size=1600, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[goal], node_color="deepskyblue",
                           edgecolors="black", node_size=1600, ax=ax)

    # 5. Labels: building names and travel times
    nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
    edge_labels = {(u, v): f"{d['weight']}s" for u, v, d in graph.edges(data=True)
                   if d["weight"] is not None}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax)

    if result["found"]:
        title = (f"UCS: {start} → {goal}   |   cost {result['cost']} s   |   "
                 f"{result['nodes_expanded']} nodes expanded")
    else:
        title = f"UCS: {start} → {goal}   |   no route found"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


# Short name so other files (experiments.py) can still call ucs(...)
ucs = uniform_cost_search
get_neighbours = successor_fn


if __name__ == "__main__":
    print("Locations:", ", ".join(sorted(G.nodes)))
    start = input("Start: ").strip().upper() or "SHA"
    goal = input("Goal:  ").strip().upper() or "BUS"

    if start not in G or goal not in G:
        print("Unknown location.")
    else:
        result = uniform_cost_search(G, start, goal)
        print(f"\nUCS: {start} -> {goal}")
        if result["found"]:
            print("  Route:         ", " -> ".join(result["path"]))
            print("  Path cost:     ", f"{result['cost']} s")
        else:
            print("  Failure: no route exists")
        print("  Nodes expanded:", result["nodes_expanded"])
        print(f"  Run time:       {result['time_ms']:.3f} ms")

        # Show the route on the campus map
        show_path(G, start, goal, result)