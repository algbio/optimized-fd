# Algorithmic Optimization applied to Minimum Flow Decomposition via Integer Linear Programming

## Introduction

In the Minimum Flow Decomposition (MFD) problem, we are given a flow in a directed acyclic graph (DAG) with unique source *s* and unique sink *t*, and we need to decompose it into the minimum number of weighted paths (usually the weights are positive integers) from *s* to *t*, such that for every edge the weights of the paths crossing the edge sum up to the flow value.

In the image below, an example of a flow network is displayed: 

![MFD Example](https://github.com/FernandoHDias/optimized-fd/raw/main/MFD-1.png) 

which generates the following decomposition into 5 paths:

![MFD Example](https://github.com/FernandoHDias/optimized-fd/raw/main/MFD-2.png) 

MFD-optimized is an upgraded tool for MFD using integer linear programming by implementing several optimization that reduce the dimension of the search space (number of unkown variables).

## Pre-requisites

### Python libraries
  - Gurobipy (version 10.0.1)
    - Activate the license as instructed in [www.gurobi.com](www.gurobi.com).
  - Networkx (version 2.4)

### Practical MPC
```TODO```

## Run

To run the formulation, use `python3` to execute the `mfd_optimization.py` file.
Run `python3 ./src/mfd_optimization.py -h` for help.

### Input graph

- Use `-i/--input FILE`, where FILE contains the input DAGs and flow values.
- The input file contains a sequence of flow networks separated by lines starting with `#`.
- The first line of each flow network contains the number of vertices of the graph
- Every following line `u v f` describes an edge from vertex `u` to  `v` carrying `f` flow.
- Vertices must be integers following a topological order of the graph.

### Heuristic solution

- To use safety optimization, a given flow decomposition is needed.
- Use `--heuristic HEURISTIC`, where HEURISTIC contains a heuristic solution for every graph in the input FILE.
- Every flow network is separated by a line starting with `#`.
- Every following line represents a path `v1,v2,...` of weight `w` in the following format:
  `w: [(v1, v2), (v2, v3), (v3, v4), ...]`.

### Inexact MFD

This tool supports inexact flow decompositions, where the input graphs are not flow networks anymore but
DAGs with intervals for every edge. An inexact flow decomposition is a flow decomposition of a feasible flow
in these intervals.

- Use `--inexact` for inexact MFDs.
- Note that the input FILE must have edges of the form `u v l r`, describing edges from vertex `u` to `v` with interval `[l,r]`.
- Note that the input FILE is not allowed to have subpath constraints with inexact MFDs.

### Subpath constraints

This tool supports subpath constraints, i.e. given subpaths `S`, it  can find the minimum flow decomposition given the restriction
that every path in `S` is a subpath of some weighted path in the MFD.

```
The rest is TODO.
```
