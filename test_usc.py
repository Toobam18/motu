"""
(FIX 6: file renamed from "test usc.py" to test_usc.py so pytest finds it)

MOTU - UCS correctness tests (progress report: "Evidence of testing")

Each test has: test case, expected, actual, pass/fail.
Expected values are hand-computed from the travel times in script.py,
or taken from networkx's Dijkstra as an independent reference
(networkx is only the answer key, never our search).
"""

import networkx as nx

from script import G, build_campus_graph, TRAVEL_TIMES
from USC import ucs   # our file is named USC.py

results = []   # rows of (test case, expected, actual, passed)


def check(name, expected, actual):
    results.append((name, str(expected), str(actual), expected == actual))


# 1. Normal route, hand-computed: ENG -> UA (240) -> BUS (300) = 540 s
r = ucs(G, "ENG", "BUS")
check("Normal route ENG -> BUS", 540, r["cost"])

# 2. Start = goal: goal test passes immediately, cost 0
r = ucs(G, "UA", "UA")
check("Start = goal (UA -> UA)", 0, r["cost"])

# 3. Unknown building name: should raise an error, not return a route
try:
    ucs(G, "XYZ", "UA")
    actual = "no error"
except ValueError:                 # FIX 7: was `except Exception`
    actual = "error raised"
check("Unknown building (XYZ)", "error raised", actual)

# 3b. Unknown goal should also raise an error   (FIX 7: new test)
try:
    ucs(G, "UA", "XYZ")
    actual = "no error"
except ValueError:
    actual = "error raised"
check("Unknown goal (XYZ)", "error raised", actual)

# 4. Cheapest path is NOT the fewest hops:
#    LIB -> UA -> UB = 180 + 120 = 300 s (2 hops)
#    LIB -> SHA -> ERC -> UB = 120 + 30 + 30 = 180 s (3 hops)  <- cheaper
r = ucs(G, "LIB", "UB")
check("Cheapest != fewest hops (LIB -> UB)", 180, r["cost"])

# 5. Edge with None weight is skipped:
#    make LIB-SHA unknown, so LIB -> SHA must go LIB-UA-UB-ERC-SHA
#    = 180 + 120 + 30 + 30 = 360 s
times = dict(TRAVEL_TIMES)
times["LIBtoSHA"] = None
H = build_campus_graph(times)
r = ucs(H, "LIB", "SHA")
check("None-weight edge skipped (LIB -> SHA)", 360, r["cost"])

# 6. Unreachable goal: remove UP's only connection -> failure
H = build_campus_graph()
H.remove_edge("REC", "UP")
r = ucs(H, "SIRC", "UP")
check("Unreachable goal (UP cut off)", False, r["found"])

# 7. Symmetry: walking A -> B costs the same as B -> A on all 45 pairs
nodes = sorted(G.nodes)
pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
mismatch = sum(ucs(G, a, b)["cost"] != ucs(G, b, a)["cost"] for a, b in pairs)
check("Symmetric cost on all 45 pairs", 0, mismatch)

# 8. Optimality: UCS cost matches Dijkstra reference on all 45 pairs
mismatch = sum(ucs(G, a, b)["cost"] != nx.dijkstra_path_length(G, a, b)
               for a, b in pairs)
check("Matches Dijkstra on all 45 pairs", 0, mismatch)


# ---------------------------------------------------------------------------
# Print the test table
# ---------------------------------------------------------------------------
print(f"{'#':<3}{'Test case':<40}{'Expected':<15}{'Actual':<15}Result")
print("-" * 80)
for i, (name, exp, act, ok) in enumerate(results, 1):
    print(f"{i:<3}{name:<40}{exp:<15}{act:<15}{'PASS' if ok else 'FAIL'}")
print("-" * 80)
print(f"{sum(r[3] for r in results)}/{len(results)} tests passed")
