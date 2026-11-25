"""
MOTU - A* and Greedy correctness tests (progress report: "Evidence of testing")

Same table format as test_usc.py. networkx's Dijkstra is only the answer key.
"""

import networkx as nx

from script import G, build_campus_graph, generate_random_graph
from USC import ucs
from astar import a_star
from greedy import greedy

results = []


def check(name, expected, actual):
    results.append((name, str(expected), str(actual), expected == actual))


def all_pairs(graph):
    ns = sorted(graph.nodes)
    return [(a, b) for i, a in enumerate(ns) for b in ns[i + 1:]]


def dijkstra_cost(graph, a, b):
    return (nx.dijkstra_path_length(graph, a, b)
            if nx.has_path(graph, a, b) else float("inf"))


# 30 random 20-node maps, 30% obstacles that may cut the map
RANDOM = [generate_random_graph(n_nodes=20, seed=s, obstacle_density=0.3,
                                keep_connected=False) for s in range(30)]
CAMPUS_PAIRS = all_pairs(G)

# ---------------------------------------------------------------- A*
# 1. Same hand-computed route as the UCS test: ENG -> UA -> BUS = 540 s
check("A* normal route ENG -> BUS", 540, a_star(G, "ENG", "BUS")["cost"])

# 2. Cheapest != fewest hops: LIB -> SHA -> ERC -> UB = 180 s
check("A* cheapest != fewest hops (LIB->UB)", "LIB-SHA-ERC-UB",
      "-".join(a_star(G, "LIB", "UB")["path"]))

# 3. Start = goal
check("A* start = goal (UA -> UA)", 0, a_star(G, "UA", "UA")["cost"])

# 4. Unknown location raises ValueError
try:
    a_star(G, "UA", "XYZ"); actual = "no error"
except ValueError:
    actual = "error raised"
check("A* unknown goal (XYZ)", "error raised", actual)

# 5. Unreachable goal -> failure
H = build_campus_graph(); H.remove_edge("REC", "UP")
check("A* unreachable goal (UP cut off)", False, a_star(H, "SIRC", "UP")["found"])

# 6. Optimal: matches Dijkstra on all 45 campus pairs
bad = sum(a_star(G, a, b)["cost"] != dijkstra_cost(G, a, b) for a, b in CAMPUS_PAIRS)
check("A* matches Dijkstra, 45 campus pairs", 0, bad)

# 7. Optimal on random maps (incl. unreachable pairs -> inf)
bad = sum(a_star(R, a, b)["cost"] != dijkstra_cost(R, a, b)
          for R in RANDOM for a, b in all_pairs(R))
check("A* matches Dijkstra, 30 random maps", 0, bad)

# 8. Informed search never expands MORE nodes than UCS (consistent h)
bad = sum(a_star(R, a, b)["nodes_expanded"] > ucs(R, a, b)["nodes_expanded"]
          for R in RANDOM for a, b in all_pairs(R))
check("A* expands <= UCS, 30 random maps", 0, bad)

# 9. h_weight = 0 turns A* into UCS (same cost on every campus pair)
bad = sum(a_star(G, a, b, h_weight=0)["cost"] != ucs(G, a, b)["cost"]
          for a, b in CAMPUS_PAIRS)
check("A* with h=0 behaves like UCS", 0, bad)

# ---------------------------------------------------------------- Greedy
# 10. Start = goal
check("Greedy start = goal (UA -> UA)", 0, greedy(G, "UA", "UA")["cost"])

# 11. Unknown location raises ValueError
try:
    greedy(G, "XYZ", "UA"); actual = "no error"
except ValueError:
    actual = "error raised"
check("Greedy unknown start (XYZ)", "error raised", actual)

# 12. Complete: finds a route exactly when one exists (random maps)
bad = sum(greedy(R, a, b)["found"] != nx.has_path(R, a, b)
          for R in RANDOM for a, b in all_pairs(R))
check("Greedy complete, 30 random maps", 0, bad)

# 13. Never cheaper than optimal (it can't beat UCS)
bad = sum(greedy(R, a, b)["cost"] < dijkstra_cost(R, a, b)
          for R in RANDOM for a, b in all_pairs(R))
check("Greedy cost >= optimal, 30 random maps", 0, bad)

# 14. Reported cost = real sum of the edges on its path
def path_cost(graph, path):
    return sum(graph[u][v]["weight"] for u, v in zip(path, path[1:]))
bad = 0
for R in RANDOM:
    for a, b in all_pairs(R):
        r = greedy(R, a, b)
        if r["found"] and r["cost"] != path_cost(R, r["path"]):
            bad += 1
check("Greedy cost = sum of its path edges", 0, bad)

# 15. Greedy is NOT optimal: at least one random pair where it pays more
worse = sum(greedy(R, a, b)["cost"] > dijkstra_cost(R, a, b)
            for R in RANDOM for a, b in all_pairs(R))
check("Greedy sometimes suboptimal (expected)", True, worse > 0)


# ---------------------------------------------------------------------------
print(f"{'#':<4}{'Test case':<42}{'Expected':<26}{'Actual':<26}Result")
print("-" * 104)
for i, (name, exp, act, ok) in enumerate(results, 1):
    print(f"{i:<4}{name:<42}{exp:<26}{act:<26}{'PASS' if ok else 'FAIL'}")
print("-" * 104)
print(f"{sum(r[3] for r in results)}/{len(results)} tests passed")
print(f"(Greedy was more expensive than optimal on {worse} of "
      f"{sum(len(all_pairs(R)) for R in RANDOM)} random pairs)")
