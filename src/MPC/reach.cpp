#include "graph.h"
#include "naive.h"
#include "reach.h"
#include "pflow.h"
#include <memory>
#include <vector>
#include <fstream> 

std::unique_ptr<reachability_idx> graph_reachability(Graph &g, const std::vector<std::tuple<int, int, int, int, int, int>> &control_reachability) {
    auto mf = pflowk2(g);
    auto pc = minflow_reduction_path_recover_faster(*mf);
    return std::make_unique<reachability_idx>(g, pc, control_reachability);
}

void save_graph_to_dot(const Graph &g, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    file << "digraph G {\n";  

    for (int i = 1; i <= g.n; ++i) {
        for (auto u : g.edge_out[i]) {
            file << "    " << i << " -> " << u << ";\n";  
        }
    }

    file << "}\n"; 
    file.close();
    std::cout << "Graph saved to " << filename << std::endl;
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

    // uncomment this part to visualize the graph
    // save_graph_to_dot(g, "graph.dot");
    // system("dot -Tpng graph.dot -o graph.png");

    auto reachability = graph_reachability(g, control_reachability);
    return 0;
}
