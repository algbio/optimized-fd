#include "graph.h"
#include "naive.h"
#include "reach.h"
#include "pflow.h"
#include <memory>
#include <vector>

std::unique_ptr<reachability_idx> graph_reachability(Graph &g, const std::vector<std::tuple<int, int, int, int, int, int>> &control_reachability) {
    auto mf = pflowk2(g);
    auto pc = minflow_reduction_path_recover_faster(*mf);
    return std::make_unique<reachability_idx>(g, pc, control_reachability);
}

int main() {
    std::ios_base::sync_with_stdio(0);
    std::cin.tie(0);

    // reading control reachability tuples:
    int control_reachability_size;
    std::cin >> control_reachability_size;

    std::vector<std::tuple<int, int, int, int, int, int>> control_reachability(control_reachability_size);
    for (int i = 0; i < control_reachability_size; ++i) {
        int u, v, start_node_of_edge, end_node_of_edge, edge_index, path_index;
        std::cin >> u >> v >> start_node_of_edge >> end_node_of_edge >> edge_index >> path_index;
        control_reachability[i] = std::make_tuple(u, v, start_node_of_edge, end_node_of_edge, edge_index, path_index);
    }

    // reading graph
    int n, m;
    std::cin >> n >> m;

    Graph g(n);
    for (int i = 0; i < m; i++) {
        int a, b;
        std::cin >> a >> b;
        a++; b++;
        g.add_edge(a, b);
    }

    auto reachability = graph_reachability(g, control_reachability);

    return 0;
}
