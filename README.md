# MOTU – Route Planner

## Setup (VS Code terminal)
    pip install -r requirements.txt

## Run the web front end
    python app.py
Then open http://127.0.0.1:5000 in your browser.

## Other entry points
    python USC.py          # UCS in the terminal + matplotlib map
    python astar.py        # A* in the terminal
    python greedy.py       # Greedy best-first in the terminal
    python test_usc.py     # UCS correctness tests (9 tests)
    python test_search.py  # A* + Greedy correctness tests (15 tests)
    python experiments.py  # proposal's experiments: 30 trials/setting -> results/
    python script.py     # print/draw a random graph

## Algorithms in the front end
app.py imports ucs (USC.py), a_star (astar.py) and greedy (greedy.py) into the
ALGORITHMS dict. All three take (graph, start, goal) and return:
found, path, cost, nodes_expanded, time_ms, explored.
Heuristic for A* / Greedy = straight-line distance * h_scale (script.py).

## Fixes applied (search the code for "# FIX")
1. script.py  – campus graph gets h_scale (30 s/unit) so A* has a useful heuristic
2. script.py  – get_map docstring example n_nodes=100 -> 25
3. script.py  – removed duplicate print in __main__
4. USC.py     – raises ValueError for unknown start/goal
5. USC.py     – show_path uses the graph's own positions (works on random graphs)
6. test usc.py renamed to test_usc.py
7. test_usc.py – test 3 catches ValueError; new test for an unknown goal
