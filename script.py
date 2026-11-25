"""
Graph environments for the AI route-planning project (MOTU).

Two kinds of environment live here:

1. The hand-built CAMPUS graph (build_campus_graph / G) - unchanged.
2. RANDOM graphs (generate_random_graph) for the experiments.

Nodes are locations; edges are paths weighted by travel cost. Everything is
undirected. networkx only STORES the graph; all generation logic is ours.

Requires: pip install networkx matplotlib
"""

import math
import random

import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CAMPUS GRAPH (hand-built map, unchanged)
# ---------------------------------------------------------------------------
TRAVEL_TIMES = {
    "SIRCtoENG": 300,
    "ENGtoLIB": 180,
    "LIBtoSHA": 120,
    "SHAtoERC": 30,
    "ERCtoUB": 30,
    "UBtoUA": 120,
    "LIBtoUA": 180,
    "ENGtoUA": 240,
    "ENGtoREC": 180,
    "RECtoUA": 180,
    "UAtoBUS": 300,
    "RECtoUP": 120,
}

# (building 1, building 2, variable name, indoor option?)
EDGES = [
    ("SIRC", "ENG", "SIRCtoENG", False),
    ("ENG",  "LIB", "ENGtoLIB", False),
    ("LIB",  "SHA", "LIBtoSHA", False),
    ("SHA",  "ERC", "SHAtoERC", True),
    ("ERC",  "UB",  "ERCtoUB", True),
    ("UB",   "UA",  "UBtoUA", False),
    ("LIB",  "UA",  "LIBtoUA", False),
    ("ENG",  "UA",  "ENGtoUA", False),
    ("ENG",  "REC", "ENGtoREC", False),
    ("REC",  "UA",  "RECtoUA", False),
    ("UA",   "BUS", "UAtoBUS", False),
    ("REC",  "UP",  "RECtoUP", False),
]

INDOOR_BUILDINGS = {"SHA", "ERC", "UB"}

# Node positions roughly matching the hand-drawn diagram.
POSITIONS = {
    "SIRC": (0, 6),
    "ENG":  (0, 4),
    "LIB":  (2, 4),
    "SHA":  (4, 4),
    "ERC":  (4, 3),
    "UB":   (4, 2),
    "UA":   (2, 1),
    "REC":  (0, 1),
    "UP":   (0, -1),
    "BUS":  (3, -1),
}


def build_campus_graph(times=None):
    """Build an undirected weighted graph of the campus.

    Edge attributes:
        var     - edge name (e.g. "ENGtoLIB")
        weight  - travel time in seconds (None if unknown)
        indoor  - True if an indoor option exists on that segment
    """
    times = TRAVEL_TIMES if times is None else times
    G = nx.Graph()

    for building in {n for u, v, _, _ in EDGES for n in (u, v)}:
        G.add_node(building, indoor_option=building in INDOOR_BUILDINGS,
                   pos=POSITIONS[building])

    for u, v, var, indoor in EDGES:
        G.add_edge(u, v, var=var, weight=times.get(var), indoor=indoor)

    G.graph["name"] = "campus"
    G.graph["positions"] = dict(POSITIONS)
    # FIX 1: seconds per map unit for the campus heuristic. The smallest
    # cost/distance ratio keeps h admissible (works out to 30 s/unit).
    # Uses math.hypot because _euclid is defined further down the file.
    G.graph["h_scale"] = min(
        d["weight"] / math.hypot(POSITIONS[u][0] - POSITIONS[v][0],
                                 POSITIONS[u][1] - POSITIONS[v][1])
        for u, v, d in G.edges(data=True) if d["weight"] is not None)
    return G


G = build_campus_graph()


# ---------------------------------------------------------------------------
# 2. RANDOM GRAPH GENERATOR
# ---------------------------------------------------------------------------
#
# Model (a randomized "road network"):
#   * n nodes are dropped at random (x, y) points in a size x size square.
#   * A minimum spanning tree (Kruskal + union-find, our own code) is built
#     first, so the graph is CONNECTED and a path always exists.
#   * Extra edges between near neighbours are then added until the requested
#     average degree is reached. Higher degree = more alternative paths.
#   * Each edge cost (seconds) = euclidean_length * secs_per_unit * detour,
#     rounded up and never below min_cost (60 s). The detour factor is >= 1
#     and drawn from a chosen distribution (see COST_DISTRIBUTIONS).
#     secs_per_unit is chosen so a typical edge is ~90 s (most are 60-180).
#     Because cost >= length * secs_per_unit, the scaled straight-line
#     distance is an ADMISSIBLE and CONSISTENT heuristic for Greedy/A*
#     (see euclidean_heuristic below).
#   * The graph is simple: at most one edge per pair of nodes, no self-loops.
#     Allowed sizes: 2 to 29 nodes.
#   * Obstacles: a fraction of edges is blocked (removed) after the fact.
#     By default we never block an edge if it would disconnect the graph;
#     set keep_connected=False to allow unreachable goals (failure cases).
#
# Everything is driven by one seed, so every experiment is reproducible.

COST_DISTRIBUTIONS = ("distance", "uniform", "skewed", "bimodal")

# Project limits: graphs are small (fewer than 30 nodes) and edge costs are
# travel times in SECONDS, typically 60-180 (long edges may exceed that).
MIN_NODES = 2
MAX_NODES = 29
MIN_COST = 60      # no edge is cheaper than this many seconds
BASE_COST = 90     # a "typical" edge (median length, no detour) costs ~this


def _detour_factor(rng, dist, spread):
    """Multiplier (>= 1) applied to an edge's straight-line length.

    distance : factor = 1 exactly. Cost IS the distance, so the heuristic is
               perfect on open ground (best case for A*).
    uniform  : factor ~ U(1, 1 + spread). Costs vary evenly.
    skewed   : factor = 1 + Exponential. Most edges are cheap, a few are very
               expensive (traffic jams / steep roads). Long tail.
    bimodal  : 70% fast roads (factor 1), 30% slow roads (factor 1 + spread
               + noise). Creates "highway vs. back road" structure that makes
               the heuristic less informative.
    """
    if dist == "distance":
        return 1.0
    if dist == "uniform":
        return rng.uniform(1.0, 1.0 + spread)
    if dist == "skewed":
        return 1.0 + rng.expovariate(1.0 / max(spread, 1e-9))
    if dist == "bimodal":
        if rng.random() < 0.7:
            return 1.0
        return 1.0 + spread + rng.uniform(0, spread * 0.25)
    raise ValueError(f"cost_dist must be one of {COST_DISTRIBUTIONS}, got {dist!r}")


def _euclid(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


class _UnionFind:
    """Tiny union-find used by Kruskal's spanning-tree step."""

    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def generate_random_graph(n_nodes=20, avg_degree=4.0, cost_dist="uniform",
                          cost_spread=1.0, obstacle_density=0.0,
                          keep_connected=True, size=100, seed=None,
                          min_cost=MIN_COST, base_cost=BASE_COST,
                          min_degree=2):
    """Generate a random, undirected, weighted graph with 2-D positions.

    Parameters (these are the experiment knobs from the project spec):
        n_nodes          size of the search space, 2 to 29 locations. The graph
                         is simple: at most ONE edge between two nodes and no
                         self-loops (so 2 nodes -> exactly 1 edge).
        avg_degree       edges per node on average. ~2 = tree-like (one or
                         few paths); 4-6 = dense with many alternative paths.
                         Clamped to [2(n-1)/n, n-1].
        cost_dist        distribution of edge costs: one of COST_DISTRIBUTIONS
        cost_spread      how strong the cost variation is (see _detour_factor)
        obstacle_density fraction (0-1) of the edges that get BLOCKED
        keep_connected   True: never block an edge that disconnects the graph
                         (a path always exists). False: blocking may cut the
                         graph, producing unreachable goals.
        size             side length of the square the nodes live in
        seed             random seed (same seed + same args = same graph)
        min_degree       every node gets at least this many edges (default 2),
                         so there are no dead-end spurs and the cheapest
                         route is rarely obvious from the picture
        min_cost         cheapest allowed edge, in seconds (default 60)
        base_cost        seconds for a typical edge before the detour factor
                         (default 90). Most costs land around 60-180 s; edges
                         that are long on the map can be larger.

    Node names are "N0", "N1", ...

    Graph attributes:
        G.graph["positions"]  {node: (x, y)}
        G.graph["h_scale"]    seconds per map unit; straight-line distance times
                              this is an admissible heuristic (in seconds)
        G.graph["params"]     the generation parameters (for experiment logs)
        G.graph["blocked"]    list of blocked edges (obstacles)
    Node attribute:  pos
    Edge attributes: weight (int cost), length (straight-line), indoor (False)
    """
    if not MIN_NODES <= n_nodes <= MAX_NODES:
        raise ValueError(f"n_nodes must be between {MIN_NODES} and {MAX_NODES}")
    if not 0 <= obstacle_density <= 1:
        raise ValueError("obstacle_density must be between 0 and 1")

    if seed is None:                       # no seed given -> pick a fresh one,
        seed = random.randrange(10**6)     # but record it so the graph can be recreated
    rng = random.Random(seed)
    names = [f"N{i}" for i in range(n_nodes)]

    # --- 1. random positions (reject points that are almost on top of
    #        each other so edge lengths are never ~0) ----------------------
    min_sep = size / (4 * math.sqrt(n_nodes))
    positions = {}
    for name in names:
        for _ in range(200):
            p = (round(rng.uniform(0, size), 2), round(rng.uniform(0, size), 2))
            if all(_euclid(p, q) >= min_sep for q in positions.values()):
                break
        positions[name] = p            # (accept the last try if it was crowded)

    # --- 2. all candidate edges, shortest first ----------------------------
    candidates = sorted(
        (_euclid(positions[a], positions[b]), a, b)
        for i, a in enumerate(names) for b in names[i + 1:]
    )

    # --- 3. spanning tree (Kruskal) => graph is connected ------------------
    uf = _UnionFind(names)
    chosen = set()
    for _, a, b in candidates:
        if uf.union(a, b):
            chosen.add((a, b))
        if len(chosen) == n_nodes - 1:
            break

    # --- 4. extra edges for density. Only consider each node's nearest
    #        neighbours so the graph stays road-like (no huge cross-map
    #        edges), then sample from them at random. ----------------------
    max_edges = n_nodes * (n_nodes - 1) // 2
    target = int(round(avg_degree * n_nodes / 2))
    target = max(n_nodes - 1, min(target, max_edges))

    k = max(4, int(avg_degree) + 3)
    near = set()
    for a in names:
        by_dist = sorted((n for n in names if n != a),
                         key=lambda n: _euclid(positions[a], positions[n]))
        for b in by_dist[:k]:
            near.add((a, b) if int(a[1:]) < int(b[1:]) else (b, a))
    pool = sorted(near - chosen)
    rng.shuffle(pool)
    for edge in pool:
        if len(chosen) >= target:
            break
        chosen.add(edge)
    if len(chosen) < target:        # tiny graphs: fall back to any pair
        rest = [(a, b) for _, a, b in candidates if (a, b) not in chosen]
        rng.shuffle(rest)
        for edge in rest:
            if len(chosen) >= target:
                break
            chosen.add(edge)

    # --- 4b. no dead ends: every node gets at least min_degree edges,
    #         connecting it to its nearest non-neighbours first ---------------
    min_degree = max(1, min(min_degree, n_nodes - 1))
    for a in names:
        if sum(a in e for e in chosen) >= min_degree:
            continue
        by_dist = sorted((n for n in names if n != a),
                         key=lambda n: _euclid(positions[a], positions[n]))
        for b in by_dist:
            if sum(a in e for e in chosen) >= min_degree:
                break
            chosen.add((a, b) if int(a[1:]) < int(b[1:]) else (b, a))

    # --- 5. build the networkx graph with costs ----------------------------
    graph = nx.Graph()
    for name in names:
        graph.add_node(name, pos=positions[name], indoor_option=False)
    # Convert map distance to seconds: the median edge length maps to
    # base_cost seconds. cost >= length * secs_per_unit always holds (the
    # detour factor is >= 1, and min_cost / ceil only round UP), so the
    # straight-line heuristic stays admissible.
    lengths = sorted(_euclid(positions[a], positions[b]) for a, b in chosen)
    median_len = lengths[len(lengths) // 2]
    secs_per_unit = base_cost / median_len
    for a, b in sorted(chosen):
        length = _euclid(positions[a], positions[b])
        factor = _detour_factor(rng, cost_dist, cost_spread)
        cost = max(min_cost, math.ceil(length * secs_per_unit * factor))
        graph.add_edge(a, b, var=f"{a}to{b}", weight=cost,
                       length=round(length, 3), indoor=False)

    # --- 6. obstacles: block a fraction of edges ---------------------------
    blocked = []
    n_block = int(round(obstacle_density * graph.number_of_edges()))
    edges = sorted(graph.edges())
    rng.shuffle(edges)
    for a, b in edges:
        if len(blocked) >= n_block:
            break
        data = graph[a][b]
        graph.remove_edge(a, b)
        if keep_connected and not nx.is_connected(graph):
            graph.add_edge(a, b, **data)    # would cut the map: keep it open
        else:
            blocked.append((a, b))

    graph.graph.update(
        name=f"random(n={n_nodes}, deg={avg_degree}, {cost_dist}, "
             f"obs={obstacle_density}, seed={seed})",
        positions=positions,
        h_scale=secs_per_unit,
        blocked=blocked,
        params=dict(n_nodes=n_nodes, avg_degree=avg_degree, cost_dist=cost_dist,
                    cost_spread=cost_spread, obstacle_density=obstacle_density,
                    keep_connected=keep_connected, size=size, seed=seed,
                    min_cost=min_cost, base_cost=base_cost,
                    min_degree=min_degree),
    )
    return graph


# ---------------------------------------------------------------------------
# 3. START / GOAL SELECTION
# ---------------------------------------------------------------------------
# Distance between start and goal is one of the experiment variables, so we
# can ask for a "near", "medium" or "far" pair (by straight-line distance as
# a fraction of the farthest pair in the graph), or just "any".

DISTANCE_BANDS = {
    "near":   (0.00, 0.25),
    "medium": (0.25, 0.60),
    "far":    (0.60, 1.00),
    "any":    (0.00, 1.00),
}


def pick_start_goal(graph, band="any", require_path=True, seed=None):
    """Pick a random (start, goal) pair whose straight-line distance falls in
    the given band ("near" / "medium" / "far" / "any").

    require_path=True only returns pairs that are actually connected.
    Falls back to the closest-fitting pair if the band is empty.
    """
    if band not in DISTANCE_BANDS:
        raise ValueError(f"band must be one of {list(DISTANCE_BANDS)}")
    rng = random.Random(seed)
    nodes = sorted(graph.nodes)
    pos = {n: graph.nodes[n]["pos"] for n in nodes}

    comp = {}
    if require_path:
        for i, c in enumerate(nx.connected_components(graph)):
            for n in c:
                comp[n] = i

    pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
             if not require_path or comp[a] == comp[b]]
    if not pairs:
        raise ValueError("no connected pair of nodes exists")

    dists = {p: _euclid(pos[p[0]], pos[p[1]]) for p in pairs}
    max_d = max(dists.values())
    lo, hi = DISTANCE_BANDS[band]
    ok = [p for p in pairs if lo * max_d <= dists[p] <= hi * max_d]
    if not ok:
        mid = (lo + hi) / 2 * max_d
        ok = [min(pairs, key=lambda p: abs(dists[p] - mid))]

    start, goal = rng.choice(ok)
    if rng.random() < 0.5:
        start, goal = goal, start
    return start, goal


# ---------------------------------------------------------------------------
# 4. HEURISTIC SUPPORT (used later by Greedy / A*)
# ---------------------------------------------------------------------------

def euclidean_heuristic(graph, node, goal, scale=None):
    """Estimated travel time (seconds) from node to goal: straight-line
    distance on the map times the graph's seconds-per-unit (h_scale).

    ADMISSIBLE: every edge costs >= its straight-line length * h_scale, so no
    path can be cheaper than the straight line. CONSISTENT: follows from the
    triangle inequality.

    `scale` is a fraction of that full heuristic, for "heuristic quality"
    experiments: 1.0 (default) = full, 0.5 = weaker, 0 = no heuristic (A*
    behaves like UCS). Values above 1 make it inadmissible.
    Only valid for graphs that carry positions (random graphs / campus).
    """
    if scale is None:
        scale = 1.0
    return (scale * graph.graph.get("h_scale", 1.0)
            * _euclid(graph.nodes[node]["pos"], graph.nodes[goal]["pos"]))


# ---------------------------------------------------------------------------
# 5. MAP PRESETS ("select an environment")
# ---------------------------------------------------------------------------
MAP_PRESETS = {
    "campus":         None,   # hand-built graph above
    "tiny":           dict(n_nodes=6,   avg_degree=3.0, cost_dist="uniform"),
    "small_sparse":   dict(n_nodes=12,  avg_degree=3.5, cost_dist="uniform"),
    "small_dense":    dict(n_nodes=12,  avg_degree=5.5, cost_dist="uniform"),
    "medium":         dict(n_nodes=20,  avg_degree=4.0, cost_dist="uniform"),
    "medium_skewed":  dict(n_nodes=20,  avg_degree=4.0, cost_dist="skewed",
                           cost_spread=1.0),
    "medium_bimodal": dict(n_nodes=20,  avg_degree=4.0, cost_dist="bimodal",
                           cost_spread=1.5),
    "medium_blocked": dict(n_nodes=20,  avg_degree=5.5, cost_dist="uniform",
                           obstacle_density=0.3),
    "large":          dict(n_nodes=29,  avg_degree=4.5, cost_dist="uniform"),
    "large_dense":    dict(n_nodes=29,  avg_degree=7.0, cost_dist="uniform"),
}


def get_map(name, seed=None, **overrides):
    """Return a map by preset name. Extra keyword args override the preset,
    e.g. get_map("medium", seed=1, n_nodes=25)."""   # FIX 2: was 100 (> MAX_NODES)
    if name not in MAP_PRESETS:
        raise ValueError(f"unknown map {name!r}; choose from {list(MAP_PRESETS)}")
    if name == "campus":
        return build_campus_graph()
    params = {**MAP_PRESETS[name], **overrides}
    return generate_random_graph(seed=seed, **params)


def describe_graph(graph):
    """One-line summary of a graph, handy for experiment logs."""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    w = [d["weight"] for _, _, d in graph.edges(data=True) if d["weight"] is not None]
    comps = nx.number_connected_components(graph)
    return (f"{graph.graph.get('name', 'graph')}: {n} nodes, {m} edges, "
            f"avg degree {2 * m / n:.2f}, cost min/mean/max = "
            f"{min(w)}/{sum(w) / len(w):.1f}/{max(w)}, "
            f"blocked {len(graph.graph.get('blocked', []))}, "
            f"components {comps}")


# ---------------------------------------------------------------------------
# 6. DISPLAY HELPERS
# ---------------------------------------------------------------------------

def print_graph(graph=G):
    """Print the graph to the terminal as an adjacency list."""
    print(f"Graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges\n")
    for node in sorted(graph.nodes):
        print(f"{node}{' (indoor option)' if graph.nodes[node].get('indoor_option') else ''}")
        for nbr in sorted(graph.neighbors(node)):
            d = graph[node][nbr]
            time = "?" if d["weight"] is None else f"{d['weight']}s"
            tag = " [indoor]" if d.get("indoor") else ""
            print(f"  -> {nbr:5} {time:>5}{tag}")


def show_graph(graph=G, save_path=None, start=None, goal=None):
    """Draw the graph with matplotlib.

    Works for the campus graph and for random graphs. Edge labels and node
    names are only drawn for small graphs (<= 40 nodes) to stay readable.
    Blocked edges (obstacles) are drawn as faint red dotted lines.
    Pass save_path to write a PNG instead of opening a window.
    """
    pos = graph.graph.get("positions") or {n: graph.nodes[n].get("pos") for n in graph.nodes}
    if any(p is None for p in pos.values()):
        pos = nx.spring_layout(graph, seed=3)

    small = graph.number_of_nodes() <= 40
    indoor = [(u, v) for u, v, d in graph.edges(data=True) if d.get("indoor")]
    outdoor = [(u, v) for u, v, d in graph.edges(data=True) if not d.get("indoor")]
    labels = {(u, v): "?" if d["weight"] is None else d["weight"]
              for u, v, d in graph.edges(data=True)}

    fig, ax = plt.subplots(figsize=(8, 9))

    blocked = graph.graph.get("blocked", [])
    if blocked:
        nx.draw_networkx_edges(nx.Graph(blocked), pos, edge_color="red",
                               style="dotted", alpha=0.35, width=1, ax=ax)

    node_size = 1400 if small else 40
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue",
                           edgecolors="black", node_size=node_size, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=outdoor,
                           width=2 if small else 0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=indoor, width=2,
                           edge_color="red", style="dashed", ax=ax)
    if small:
        nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels,
                                     font_color="green", font_size=8, ax=ax)
    if start is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[start], node_color="limegreen",
                               edgecolors="black", node_size=node_size * 1.3, ax=ax)
    if goal is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[goal], node_color="deepskyblue",
                               edgecolors="black", node_size=node_size * 1.3, ax=ax)

    ax.set_title(graph.graph.get("name", "graph") + "  (edge labels = travel cost)")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    print(describe_graph(G))
    print()
    # FIX 3: removed a duplicate print(describe_graph(G))
    # Every setting is listed below so you can play with them. The values shown
    # are those of the "small_sparse" preset (any you delete fall back to the
    # preset's value). Same seed + same settings = the same graph every time.
    rg = get_map(
        "small_sparse",         # preset: tiny, small_sparse, small_dense, medium, medium_skewed,
                                #         medium_bimodal, medium_blocked, large, large_dense, campus
        seed=10,                # any integer = a different graph; remove it for a new graph each run
        n_nodes=12,             # number of locations, 2-29
        avg_degree=3.5,         # average edges per node: ~2.5 sparse ... 7 dense
        min_degree=2,           # fewest edges any node may have (1 allows dead ends)
        cost_dist="uniform",    # "distance", "uniform", "skewed" or "bimodal"
        cost_spread=1.0,        # strength of the cost variation (higher = bigger differences)
        obstacle_density=0.0,   # fraction of edges blocked, 0.0-1.0
        keep_connected=True,    # False lets obstacles cut the map (goal may be unreachable)
        size=100,               # side of the square the nodes are placed in (mostly cosmetic)
        min_cost=60,            # cheapest allowed edge, in seconds
        base_cost=90,           # typical edge cost in seconds (shifts the whole cost range)
    )
    print(describe_graph(rg))
    s, g = pick_start_goal(
        rg,
        band="far",             # start-goal distance: "near", "medium", "far" or "any"
        require_path=True,      # False also allows pairs with no route
        seed=20,                # only changes which start/goal pair is picked, not the graph
    )
    print(f"Example start/goal (far): {s} -> {g}")
    show_graph(rg, start=s, goal=g)