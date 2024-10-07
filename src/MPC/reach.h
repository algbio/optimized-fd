class reachability_idx {
public:
    std::vector<std::vector<int>> l2r;
    std::vector<int> some_path;
    
    reachability_idx(Graph &g, path_cover &pc, const std::vector<std::tuple<int, int, int, int, int, int>> &control_reachability) {
        l2r.resize(g.n + 1);
        for (int i = 1; i <= g.n; i++)
            l2r[i].resize(pc.size());
        some_path.resize(g.n + 1);
        std::vector<int> topo, lvl(g.n + 1);
        std::vector<bool> visited(g.n + 1);
        std::vector<std::vector<int>> rtopo_edges(g.n + 1);
        int lc = 1;
        
        auto dfs = [&lc, &g, &visited, &topo, &lvl](auto &dfs, int s) {
            if (visited[s])
                return;
            visited[s] = true;

            for (auto &u : g.edge_out[s]) {
                dfs(dfs, u);
            }

            lvl[s] = lc++;
            topo.push_back(s);
        };

        for (int i = 1; i <= g.n; i++)
            dfs(dfs, i);

        for (int i = 1; i <= g.n; i++)
            visited[i] = false;

        auto dfs2 = [&g, &visited, &rtopo_edges, &lvl](auto &dfs, int s) {
            if (visited[s])
                return;
            visited[s] = true;
            for (auto &u : g.edge_in[s]) {
                dfs(dfs, u);
            }
            for (auto &u : g.edge_in[s]) {
                rtopo_edges[u].push_back(s);
            }
        };

        for (int i = 1; i <= g.n; i++)
            dfs2(dfs2, i);

        std::vector<std::vector<int>> pth(g.n + 1);

        for (int i = 0; i < ((int) pc.size()); i++) {
            auto &path = pc[i];

            for (auto &u : path) {
                some_path[u] = i;
                pth[u].push_back(i);
            }
        }

        for (auto &u : topo) {
            for (auto &v : rtopo_edges[u]) {
                auto &p = some_path[v];

                if (l2r[u][p] >= l2r[v][p])
                    continue;

                for (int i = 0; i < ((int) pc.size()); i++) {
                    l2r[u][i] = std::max(l2r[u][i], l2r[v][i]);
                }
            }

            for (auto &i : pth[u]) {
                l2r[u][i] = lvl[u];
            }
        }

        for (const auto &cr : control_reachability) {
            int u, v, start_node_of_edge, end_node_of_edge, edge_index, path_index;
            std::tie(u, v, start_node_of_edge, end_node_of_edge, edge_index, path_index) = cr;
            std::cout << start_node_of_edge << "," << end_node_of_edge << "," << edge_index << "," << path_index << ","
                    << (reaches(u+1, v+1) ? 1 : 0) << "\n";
        }
    }

    bool reaches(int a, int b) {
        return l2r[a][some_path[b]] >= l2r[b][some_path[b]];
    }
};

std::unique_ptr<reachability_idx> graph_reachability(Graph &g, const std::vector<std::tuple<int, int, int, int, int, int>> &control_reachability);