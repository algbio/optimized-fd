Note: This README is outdated, will be updated soon.

# Algorithmic Optimization applied to Minimum Flow Decomposition via Integer Linear Programming


In the Minimul Flow Decomposition (MFD) problem, we are given a flow in a directed acyclic graph (DAG) with unique source *s* and unique sink *t*, and we need to decompose it into the minimum number of weighted paths (usually the weights are positive integers) from *s* to *t*, such that the weights of the paths sum up to the flow values, for every edge. Additional informating regarding pre-existing paths required to be in the final solution can be used in order to speed up the solution process.

In the image below, an example of a flow network is displayed: 

![MFD Example](https://github.com/FernandoHDias/optimized-fd/raw/main/MFD-1.png) 

which generates the following decomposition into 5 paths:

![MFD Example](https://github.com/FernandoHDias/optimized-fd/raw/main/MFD-2.png) 

MFD-optimized is an upgraded tool for minimum flow decompositions (mfd) using integer linear programming by implementing several optimization to reduce their size (number of variables/constrains and feasible region dimension).

An auxiliar tool for pre-processing an initial lowe bound can be in MFD-Relaxed.

# Pre-requisites

All the formulations available require the Gurobi solver to solve its model.  
Download the solver from [www.gurobi.com](www.gurobi.com), activate the (academic) license as instructed, and then install the Python API with:

```
pip3 install gurobipy
```

Also, it requires a few extra Python libraries:

  - itertools
  - more_itertools
  - math
  - os 
  - networkx 

# Run

To run the formulation, use 'python' to execute the 'mfd_optimization.py' file.

As an example you can try:

`python ./mfd_optimization/mfd_optimization.py -i ./example_inputs/example.graph -safe ./example_inputs/example.paths -o ./example_inputs/results.path`

## Input

- The input is a file containing a sequence of (directed) acyclic flow graphs separated by lines starting with `#`.
- The first line of each flow graph contains the number of vertices of the graph, after this every flow edge from vertex
`u` to  `v` carrying `f` flow is represented in a separated line in the format `u v f`.
- Vertices must be integers following a topological order of the graph.
- An example of such a format can be found in `./example_inputs/example.graph`.

## Safe 

- The safe file is an auxiliary file containining safe paths for each corresponding graph in the input file.
- The first element of each line is '-1' and the following elements are the vertices creating such path.
- Vertices must be integers following a topological order of the graph.
- An example of such a format can be found in `./example_inputs/example.paths`.

## Output

- The output is a file containing a sequence of paths separated by lines starting with `#` (one per flow
graph in the input).
- Each line contains the weight associated and the content of a path corresponding sequence of vertices.
- An example of such a format can be found in `./example_inputs/example.out`.

## Parameters

- `-i <path to input file>`. Mandatory.
- `-o <path to locate output>`. Mandatory.
- `-safe <path to safe file>`. Mandatory.

Additional Parameters (optional)
- `-stats` Output stats to file <output>.stats
- `-t <n>` Use n threads for the Gurobi solver; use 0 for all threads (default 0).
- `-ilptb <n>` Maximum time (in seconds) that the ilp solver is allowed to take when computing safe paths for one flow graph.
If the solver takes more than n seconds, then safe for (all) flow decompositions is reported instead.
- `-uef` Uses excess flow to save ILP calls.
- `-uy2v` Use Y2V contraction on the flow graphs to reduce the ILP size.
- `-s/es/rs/ess/esl/rss/rsl {scan, bin_search, exp_search, rep_exp_search}` When running the two-finger algorithm applied
the specified strategy to extend/reduce the current safe interval.
- `-st/est/rst <n>` When running the two-finger algorithm run the `small strategy` when the search space is less than n
and the `large strategy` otherwise.
- `-ugtd/-ugbu` Run a group testing algorithm (top down or bottom up) instead of two-finger.

