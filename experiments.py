"""
MOTU - Experiments (Experimental Plan + Evaluation Plan from our proposal)
=========================================================================

Proposal, "Experimental Plan" -> what we vary:
    1. Search space size          -> n_nodes          (6, 12, 20, 29)
    2. Path / obstacle density    -> obstacle_density (0%, 15%, 30%, 45% of edges blocked)
    3. Cost distribution          -> cost_dist        (distance, uniform, skewed, bimodal)
                                     ("crowded variance" multiplier on travel times)
    4. Start-to-goal distance     -> band             (near, medium, far)
    5. (bonus) Heuristic strength -> A* h_weight      (0, 0.25, 0.5, 0.75, 1.0)
       # NEW: extended to 1.5, 2.0, 3.0 (overestimating h -> A* can lose optimality)
    2b. Path density              -> avg_degree       (2.5, 4.0, 5.5, 7.0 roads per location)  # NEW: spec asks for "number/density of available paths"
    6. Failure cases              -> obstacle_density (0.3, 0.5, 0.7) with keep_connected=False  # NEW: map may be cut, so some goals are unreachable

Proposal, "Evaluation Plan" -> what we measure:
    Key metrics:  total path cost, nodes expanded, execution time,
                  success rate, optimality
    Method:       30+ random trials per setting, report the AVERAGE
    Correctness:  A* and UCS must always return the same path cost

How one trial works:
    seed -> generate_random_graph(seed)            (script.py, same seed = same map)
         -> pick_start_goal(band, seed)            (script.py)
         -> run UCS, A*, Greedy on that SAME map and SAME start/goal
    UCS is optimal, so its cost is the reference for "optimal" and "% over optimal".

Output (in the results/ folder):
    raw_results.csv    one row per (experiment, setting, trial, algorithm)
    summary.csv        the averaged tables that are printed below
    *.png              one chart per experiment, for the report
                       # NEW: + 5b_heuristic_optimality, 6_paths_nodes_expanded, 7_failure_success_rate

Run:  python experiments.py        (takes ~10-20 seconds)
"""

import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")                 # save PNGs, don't open windows
import matplotlib.pyplot as plt

from script import generate_random_graph, pick_start_goal
from USC import ucs
from astar import a_star
from greedy import greedy

TRIALS = 30                           # proposal: "30+ random trials"
RESULTS_DIR = "results"

ALGORITHMS = {"UCS": ucs, "A*": a_star, "Greedy": greedy}

# Base map used by every experiment; each experiment changes ONE knob only,
# so any difference in the results comes from that knob.
BASE = dict(n_nodes=20, avg_degree=4.0, cost_dist="uniform",
            cost_spread=1.0, obstacle_density=0.0)
BASE_BAND = "any"

# Colours: one per algorithm, the same in every chart.
COLORS = {"UCS": "#2a78d6", "A*": "#eb6834", "Greedy": "#1baf7a"}


# ---------------------------------------------------------------------------
# 1. Running trials
# ---------------------------------------------------------------------------
def run_trial(graph_params, band, seed, algorithms=ALGORITHMS):
    """One trial: build one map, pick one start/goal, run every algorithm.
    Returns {algorithm name: result dict}."""
    graph = generate_random_graph(seed=seed, **graph_params)
    # NEW: only force a connected start/goal pair when the map is kept connected.
    # NEW: with keep_connected=False the goal may be unreachable -> failure cases.
    require_path = graph_params.get("keep_connected", True)  # NEW
    start, goal = pick_start_goal(graph, band=band, seed=seed,
                                  require_path=require_path)  # NEW: pass require_path through
    return {name: fn(graph, start, goal) for name, fn in algorithms.items()}


def run_experiment(name, variable, values, make_params, raw_rows):
    """Run TRIALS trials for every value of one variable.

    make_params(value) -> (graph_params, band) for that setting.
    Appends one raw row per (setting, trial, algorithm) to raw_rows.
    """
    for value in values:
        graph_params, band = make_params(value)
        for seed in range(TRIALS):
            results = run_trial(graph_params, band, seed)
            optimal = results["UCS"]["cost"]          # UCS is optimal
            for algo, r in results.items():
                raw_rows.append({
                    "experiment": name,
                    "variable": variable,
                    "value": value,
                    "trial": seed,
                    "algorithm": algo,
                    "found": r["found"],
                    "cost": r["cost"],
                    "optimal_cost": optimal,
                    "nodes_expanded": r["nodes_expanded"],
                    "time_ms": r["time_ms"],
                })


def run_heuristic_experiment(raw_rows, weights=(0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):  # NEW: added 1.5, 2.0, 3.0
    """Bonus: same maps, only A*, with a weaker or stronger heuristic.
    h_weight = 0 means h(n) = 0, so A* becomes UCS (lecture, Ch. 4).
    NEW: h_weight > 1 OVERESTIMATES (not admissible), so A* may return a
    NEW: non-optimal route; compare optimal_% below vs above 1.0."""
    for w in weights:
        for seed in range(TRIALS):
            graph = generate_random_graph(seed=seed, **BASE)
            start, goal = pick_start_goal(graph, band=BASE_BAND, seed=seed)
            optimal = ucs(graph, start, goal)["cost"]
            r = a_star(graph, start, goal, h_weight=w)
            raw_rows.append({
                "experiment": "heuristic", "variable": "h_weight", "value": w,
                "trial": seed, "algorithm": "A*", "found": r["found"],
                "cost": r["cost"], "optimal_cost": optimal,
                "nodes_expanded": r["nodes_expanded"], "time_ms": r["time_ms"],
            })


# ---------------------------------------------------------------------------
# 2. Averaging (the Evaluation Plan metrics)
# ---------------------------------------------------------------------------
def summarize(raw_rows):
    """Average the 30 trials of every (experiment, value, algorithm)."""
    groups = {}
    for row in raw_rows:
        key = (row["experiment"], row["variable"], row["value"], row["algorithm"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (exp, var, value, algo), rows in groups.items():
        found = [r for r in rows if r["found"]]
        optimal = [r for r in found if r["cost"] == r["optimal_cost"]]
        summary.append({
            "experiment": exp,
            "variable": var,
            "value": value,
            "algorithm": algo,
            "trials": len(rows),
            "success_rate_%": round(100 * len(found) / len(rows), 1),
            "optimal_%": round(100 * len(optimal) / len(found), 1) if found else 0,
            # NEW: "if found else 0" -> no crash when every trial in a setting fails.
            # NEW: only found routes are averaged, so an inf cost (no route) is never averaged.
            "avg_cost_s": round(statistics.mean(r["cost"] for r in found), 1) if found else 0,  # NEW
            "avg_over_optimal_%": round(statistics.mean(
                100 * (r["cost"] / r["optimal_cost"] - 1) if r["optimal_cost"] else 0
                for r in found), 1) if found else 0,  # NEW
            "avg_nodes_expanded": round(statistics.mean(r["nodes_expanded"] for r in rows), 2),
            "avg_time_ms": round(statistics.mean(r["time_ms"] for r in rows), 4),
        })
    return summary


def print_table(summary, experiment, title):
    rows = [s for s in summary if s["experiment"] == experiment]
    print(f"\n{title}")
    print(f"{'value':<10}{'algorithm':<9}{'success':>9}{'optimal':>9}"
          f"{'avg cost':>11}{'over opt':>10}{'expanded':>10}{'time ms':>10}")
    print("-" * 78)
    for s in rows:
        print(f"{str(s['value']):<10}{s['algorithm']:<9}"
              f"{s['success_rate_%']:>8}%{s['optimal_%']:>8}%"
              f"{s['avg_cost_s']:>10}s{s['avg_over_optimal_%']:>9}%"
              f"{s['avg_nodes_expanded']:>10}{s['avg_time_ms']:>10.3f}")


def correctness_check(raw_rows):
    """Proposal: 'Ensure that A* and UCS always return the same path cost'."""
    by_trial = {}
    for r in raw_rows:
        if r["experiment"] == "heuristic":
            continue
        key = (r["experiment"], r["value"], r["trial"])
        by_trial.setdefault(key, {})[r["algorithm"]] = r["cost"]
    # NEW: unreachable goals give cost inf for both; inf == inf is True, so they count as a match.
    mismatches = sum(t["A*"] != t["UCS"] for t in by_trial.values())
    print(f"\nCorrectness check: A* cost == UCS cost on "
          f"{len(by_trial) - mismatches}/{len(by_trial)} trials "
          f"({'PASS' if mismatches == 0 else 'FAIL'})")


# ---------------------------------------------------------------------------
# 3. Charts for the report
# ---------------------------------------------------------------------------
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e")


def _values(summary, experiment, algo, metric):
    rows = [s for s in summary if s["experiment"] == experiment and s["algorithm"] == algo]
    return [s["value"] for s in rows], [s[metric] for s in rows]


def line_chart(summary, experiment, metric, title, xlabel, ylabel, filename):
    """One line per algorithm, labelled at the end of the line."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for algo in ALGORITHMS:
        xs, ys = _values(summary, experiment, algo, metric)
        ax.plot(xs, ys, color=COLORS[algo], linewidth=2, marker="o",
                markersize=6, label=algo)
        ax.annotate(algo, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color="#0b0b0b", fontsize=10)
    ax.set_xticks(xs)
    ax.legend(frameon=False, loc="upper left")
    _style(ax, title, xlabel, ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def grouped_bar_chart(summary, experiment, metric, title, xlabel, ylabel, filename,
                      algorithms=tuple(ALGORITHMS)):
    """One group of bars per setting, one bar per algorithm."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels, _ = _values(summary, experiment, algorithms[0], metric)
    width = 0.8 / len(algorithms)
    for i, algo in enumerate(algorithms):
        _, ys = _values(summary, experiment, algo, metric)
        xs = [j + (i - (len(algorithms) - 1) / 2) * width for j in range(len(labels))]
        bars = ax.bar(xs, ys, width=width - 0.03, color=COLORS[algo], label=algo)
        ax.bar_label(bars, fmt="%g", fontsize=8, color="#52514e", padding=2)
    ax.set_xticks(range(len(labels)), [str(v) for v in labels])
    top = max(s[metric] for s in summary if s["experiment"] == experiment)
    ax.set_ylim(0, top * 1.3)                 # headroom so the legend never hides a bar
    if len(algorithms) > 1:
        ax.legend(frameon=False, loc="upper left", ncol=len(algorithms))
    _style(ax, title, xlabel, ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def make_charts(summary):
    line_chart(summary, "size", "avg_nodes_expanded",
               "Nodes expanded vs. search space size",
               "Number of locations (n_nodes)", "Avg nodes expanded",
               "1_size_nodes_expanded.png")
    grouped_bar_chart(summary, "obstacles", "avg_nodes_expanded",
                      "Nodes expanded vs. obstacle density",
                      "Fraction of edges blocked", "Avg nodes expanded",
                      "2_obstacles_nodes_expanded.png")
    grouped_bar_chart(summary, "cost_dist", "avg_over_optimal_%",
                      "Greedy: how far above the optimal cost?",
                      "Cost distribution", "Avg % above optimal cost",
                      "3_cost_dist_greedy_gap.png", algorithms=("Greedy",))
    grouped_bar_chart(summary, "distance", "avg_nodes_expanded",
                      "Nodes expanded vs. start-to-goal distance",
                      "Start-to-goal distance band", "Avg nodes expanded",
                      "4_distance_nodes_expanded.png")

    # bonus: heuristic strength (A* only)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs, ys = _values(summary, "heuristic", "A*", "avg_nodes_expanded")
    ax.plot(xs, ys, color=COLORS["A*"], linewidth=2, marker="o", markersize=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:g}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color="#52514e")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])                  # NEW: "0.25" not "0.250" etc.
    _admissible_line(ax)                                        # NEW: mark where h stops being admissible
    ax.set_ylim(0, max(ys) * 1.25)                              # NEW: headroom so value labels clear the dashed-line label
    _style(ax, "A*: stronger heuristic = fewer nodes expanded",
           "Heuristic weight (0 = UCS, 1 = full straight-line, >1 = overestimates)",  # NEW: label covers >1
           "Avg nodes expanded")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "5_heuristic_strength.png"), dpi=150)
    plt.close(fig)

    # NEW: separate chart (no second y-axis): does A* still find the optimal route?
    fig, ax = plt.subplots(figsize=(7, 4.2))                    # NEW
    xs, ys = _values(summary, "heuristic", "A*", "optimal_%")   # NEW
    ax.plot(xs, ys, color=COLORS["A*"], linewidth=2, marker="o", markersize=6)  # NEW
    for x, y in zip(xs, ys):                                    # NEW: value label above each point
        ax.annotate(f"{y:g}%", (x, y), xytext=(0, 8), textcoords="offset points",  # NEW
                    ha="center", fontsize=9, color="#52514e")   # NEW
    ax.set_xticks(xs)                                           # NEW
    ax.set_xticklabels([f"{x:g}" for x in xs])                  # NEW
    ax.set_ylim(0, 125)                                         # NEW: room for the 100% labels + dashed-line label
    _admissible_line(ax)                                        # NEW
    _style(ax, "A*: overestimating heuristic loses optimality",  # NEW
           "Heuristic weight (0 = UCS, 1 = full straight-line, >1 = overestimates)",  # NEW
           "Routes that are optimal (%)")                       # NEW
    fig.tight_layout()                                          # NEW
    fig.savefig(os.path.join(RESULTS_DIR, "5b_heuristic_optimality.png"), dpi=150)  # NEW
    plt.close(fig)                                              # NEW

    # NEW: Experiment 2b chart - path density
    grouped_bar_chart(summary, "paths", "avg_nodes_expanded",   # NEW
                      "Nodes expanded vs. path density",        # NEW
                      "Average roads per location (avg_degree)", "Avg nodes expanded",  # NEW
                      "6_paths_nodes_expanded.png")             # NEW
    # NEW: Experiment 6 chart - failure cases
    grouped_bar_chart(summary, "failure", "success_rate_%",     # NEW
                      "Success rate when obstacles can cut the map",  # NEW
                      "Fraction of edges blocked (keep_connected=False)",  # NEW
                      "Trials where a route was found (%)",     # NEW
                      "7_failure_success_rate.png")             # NEW


def _admissible_line(ax):  # NEW: helper shared by both heuristic charts
    """NEW: dashed vertical line at h_weight = 1.0 (admissible on the left,
    overestimating on the right)."""
    ax.axvline(1.0, color="#52514e", linestyle="--", linewidth=1)  # NEW
    ax.annotate("admissible | overestimates", (1.0, 1.0),       # NEW: x in data units, y at top of axes
                xycoords=("data", "axes fraction"), xytext=(0, -2),  # NEW
                textcoords="offset points", ha="center", va="top",  # NEW
                fontsize=9, color="#52514e",                    # NEW
                bbox=dict(facecolor="white", edgecolor="none", pad=1))  # NEW: readable over grid lines


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw = []

    # Each lambda returns (graph params, start-goal band) for one setting.
    run_experiment("size", "n_nodes", [6, 12, 20, 29],
                   lambda v: ({**BASE, "n_nodes": v}, BASE_BAND), raw)
    run_experiment("obstacles", "obstacle_density", [0.0, 0.15, 0.3, 0.45],
                   lambda v: ({**BASE, "avg_degree": 5.5, "obstacle_density": v},
                              BASE_BAND), raw)
    # NEW: Experiment 2b - vary the number of roads per location (path density)
    run_experiment("paths", "avg_degree", [2.5, 4.0, 5.5, 7.0],  # NEW
                   lambda v: ({**BASE, "avg_degree": v}, BASE_BAND), raw)  # NEW
    run_experiment("cost_dist", "cost_dist", ["distance", "uniform", "skewed", "bimodal"],
                   lambda v: ({**BASE, "cost_dist": v,
                               "cost_spread": 1.5 if v == "bimodal" else 1.0},
                              BASE_BAND), raw)
    run_experiment("distance", "band", ["near", "medium", "far"],
                   lambda v: (BASE, v), raw)
    # NEW: Experiment 6 - failure cases: obstacles may cut the map apart
    run_experiment("failure", "obstacle_density", [0.3, 0.5, 0.7],  # NEW
                   lambda v: ({**BASE, "avg_degree": 3.0, "obstacle_density": v,  # NEW
                               "keep_connected": False}, BASE_BAND), raw)  # NEW
    run_heuristic_experiment(raw)

    summary = summarize(raw)

    print(f"MOTU experiments: {TRIALS} trials per setting, base map = {BASE}")
    print_table(summary, "size",      "EXPERIMENT 1 - Search space size (n_nodes)")
    print_table(summary, "obstacles", "EXPERIMENT 2 - Obstacle density (fraction of edges blocked)")
    print_table(summary, "paths",     "EXPERIMENT 2b - Path density (avg_degree = roads per location)")  # NEW
    print_table(summary, "cost_dist", "EXPERIMENT 3 - Cost distribution")
    print_table(summary, "distance",  "EXPERIMENT 4 - Start-to-goal distance")
    print_table(summary, "failure",   "EXPERIMENT 6 - Failure cases (map may be cut apart)")  # NEW
    print_table(summary, "heuristic", "BONUS - A* heuristic strength (h_weight)")
    correctness_check(raw)

    with open(os.path.join(RESULTS_DIR, "raw_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw[0].keys())
        writer.writeheader()
        writer.writerows(raw)
    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    make_charts(summary)
    print(f"\nSaved raw_results.csv, summary.csv and 8 charts to {RESULTS_DIR}/")  # NEW: 5 -> 8 charts


if __name__ == "__main__":
    main()
