#include "naive.h"
#include "graph.h"
#include "antichain.h"
#include <cassert>
#include <algorithm>
#include <array>
#include <vector>
#include <queue>
#include <stack>
#include <functional>
#include <limits>
#include <utility>
#include <iostream>
#include <memory>

// Explanation of reduction here
// Input graph should have valid and satisfied minflow
void minflow_maxflow_reduction(Flowgraph<Edge::Minflow> &fg, std::function<void(Flowgraph<Edge::Maxflow>&)> maxflow_solver) {
	assert(is_valid_minflow(fg));
	Flowgraph<Edge::Maxflow> fg_red(fg.n, fg.source, fg.sink);
	struct reduction_edge {
		Edge::Maxflow &e_reduction;
		Edge::Minflow &e_original;
		bool reverse;
	};
	std::vector<reduction_edge> v;
	int flow = 0;
	for(auto &[u,e]:fg.edge_out[fg.source])
		flow += e->flow;
	for(int i=1; i<=fg.n; i++) {
		for(auto &[u,e]:fg.edge_out[i]) {
			if(e->flow > e->demand) {
				auto *e2 = fg_red.add_edge(i, u);
				e2->capacity = e->flow - e->demand;
				v.push_back({*e2, *e, false});
			}
		}
	}
	for(int i=1; i<=fg.n; i++) {
		for(auto &[u,e]:fg.edge_out[i]) {
			auto e2 = fg_red.add_edge(u, i);
			e2->capacity = flow;
			v.push_back({*e2, *e, true});
		}
	}
	maxflow_solver(fg_red);
	for(auto &u:v) {
		if(u.reverse) {
			u.e_original.flow += u.e_reduction.flow;
		} else {
			u.e_original.flow -= u.e_reduction.flow;
		}
	}
}

// Dinitz’ Algorithm: The Original Version and Even’s Version 233
// Implementation of DA by Cherkassky
void maxflow_solve_edmonds_karp_DMOD(Flowgraph<Edge::Maxflow> &fg) {
	while(true) {
		std::vector<int> vis(fg.n+1, 0);
		std::vector<int> dist(fg.n+1, std::numeric_limits<int>::max());
		std::queue<int> q;
		q.push(fg.sink);
		dist[fg.sink] = 0;
		vis[fg.sink] = 1;
		while(!q.empty()) {
			int cur = q.front();
			q.pop();
			for(auto &[u,e]:fg.edge_out[cur]) {
				if(e->flow == 0 || vis[u])
					continue;
				vis[u] = 1;
				dist[u] = dist[cur]+1;
				q.push(u);
			}
			for(auto &[u,e]:fg.edge_in[cur]) {
				if(e->flow == e->capacity || vis[u])
					continue;
				vis[u] = 1;
				dist[u] = dist[cur]+1;
				q.push(u);
			}
		}
		if(dist[fg.source] == std::numeric_limits<int>::max()) {
			break;
		}
		std::fill(vis.begin(), vis.end(), 0);
		struct pe {
			Edge::Maxflow *e;
			bool reverse;
		};
		std::vector<pe> path;
		auto dfs = [&vis, &fg, &dist, &path](auto dfs, int s)->bool {
			if(vis[s])
				return false;
			vis[s] = 1;
			for(auto &[u,e]:fg.edge_out[s]) {
				if(e->capacity <= e->flow || dist[s]-1 != dist[u])
					continue;
				path.push_back({e, false});
				if(dfs(dfs, u))
					return true;
				path.pop_back();
			}
			for(auto &[u,e]:fg.edge_in[s]) {
				if(e->flow == 0 || dist[s]-1 != dist[u])
					continue;
				path.push_back({e, true});
				if(dfs(dfs, u))
					return true;
				path.pop_back();
			}
			if(s == fg.sink) {
				int e = std::numeric_limits<int>::max();
				for(auto u:path) {
					if(u.reverse)
						e = std::min(u.e->flow, e);
					else
						e = std::min(u.e->capacity-u.e->flow, e);
				}
				for(auto u:path) {
					if(u.reverse)
						u.e->flow -= e;
					else
						u.e->flow += e;
				}
				return true;
			}
			return false;
		};
		while(dfs(dfs, fg.source)) {
			std::fill(vis.begin(), vis.end(), 0);
			path.clear();
		}
	}
}

void maxflow_solve_edmonds_karp(Flowgraph<Edge::Maxflow> &fg) {
	while(true) {
		std::vector<std::pair<int,Edge::Maxflow*>> visited(fg.n+1);
		std::queue<int> q;
		q.push(fg.source);
		while(!q.empty()) {
			int cur = q.front();
			q.pop();
			if(cur == fg.sink) {
				break;
			}
			for(auto &[u,e]:fg.edge_out[cur]) {
				if(e->flow >= e->capacity || visited[u].first)
					continue;
				visited[u].first = cur;
				visited[u].second = e;
				q.push(u);
			}
			for(auto &[u,e]:fg.edge_in[cur]) {
				if(e->flow == 0 || visited[u].first)
					continue;
				visited[u].first = -cur;
				visited[u].second = e;
				q.push(u);
			}
		}
		if(!visited[fg.sink].first)
			break;
		int cur = fg.sink;
		int delta_flow = std::numeric_limits<int>::max();
		while(cur != fg.source) {
			if(visited[cur].first > 0) {
				delta_flow = std::min(delta_flow, visited[cur].second->capacity-visited[cur].second->flow);
			} else {
				delta_flow = std::min(delta_flow, visited[cur].second->flow);
			}
			cur = abs(visited[cur].first);
		}
		cur = fg.sink;
		while(cur != fg.source) {
			if(visited[cur].first > 0) {
				visited[cur].second->flow += delta_flow;
			} else {
				visited[cur].second->flow -= delta_flow;
			}
			cur = abs(visited[cur].first);
		}
	}
}

path_cover minflow_reduction_path_recover_faster(Flowgraph<Edge::Minflow> &fg) {
	auto v_r = [](int v){return (v+1)/2;}; // fg -> original graph
	std::vector<int> visited(fg.n+1);
	std::vector<std::vector<int>> cover;
	std::vector<std::vector<std::pair<int, Edge::Minflow*>>::iterator> edge_ptr(fg.n+1);
	for(int i=1; i<=fg.n; i++)
		edge_ptr[i] = fg.edge_out[i].begin();
	while(true) {
		std::fill(visited.begin(), visited.end(), 0);
		std::vector<int> path;
		auto dfs = [&fg, &visited, &path, &edge_ptr](auto dfs, int s)->bool {
			if(s == fg.sink) 
				return true;
			visited[s] = 1;
			while(edge_ptr[s] < fg.edge_out[s].end()) {
				auto &[u,e] = *(edge_ptr[s]);
				if(visited[u] || e->flow == 0) {
					edge_ptr[s]++;
					continue;
				}
				if(dfs(dfs, u)) {
					e->flow--;
					path.push_back(u);
					return true;
				}
			}
			return false;
		};
		if(!dfs(dfs, fg.source))
			break;
		path.push_back(fg.source);
		std::vector<int> real_path;
		for(int i=1; i<path.size()-1; i+=2) {
			real_path.push_back(v_r(path[i]));
		}
		std::reverse(real_path.begin(), real_path.end());
		cover.push_back(real_path);
	}

	for(int i=1; i<=fg.n; i++) {
		for(auto &[u, e]:fg.edge_out[i])
			assert(e->flow == 0);
	}
	return cover;
}

path_cover minflow_reduction_path_recover_fast(Flowgraph<Edge::Minflow> &fg) {
	std::vector<int> visited(fg.n+1);
	std::vector<std::stack<std::vector<int>*>> stk(fg.n+1);
	auto dfs = [&stk, &fg, &visited](auto dfs, int s) {
		if(visited[s])
			return;
		visited[s] = 1;
		for(auto &[u, e]:fg.edge_out[s]) {
			dfs(dfs, u);
		}
		for(auto &[u, e]:fg.edge_in[s]) {
			while(e->flow) {
				e->flow--;
				if(s == fg.sink) {
					auto v = new std::vector<int>();
					v->push_back(s);
					stk[s].push(v);
				}
				stk[s].top()->push_back(u);
				stk[u].push(stk[s].top());
				stk[s].pop();
			}
		}
	};
	dfs(dfs, fg.source);
	auto v_r = [](int v){return (v+1)/2;}; // fg -> original graph
	path_cover cover;
	while(!stk[fg.source].empty()) {
		auto path = stk[fg.source].top();
		stk[fg.source].pop();
		std::vector<int> real_path;
		for(int i=1; i<path->size()-1; i+=2) {
			real_path.push_back(v_r((*path)[i]));
		}
		std::reverse(real_path.begin(), real_path.end());
		cover.push_back(real_path);
		delete path;
	}
	for(int i=1; i<=fg.n; i++) {
		for(auto &[u, e]:fg.edge_out[i])
			assert(e->flow == 0);
	}
	return cover;
}

path_cover minflow_reduction_path_recover(Flowgraph<Edge::Minflow> &fg) {
	auto v_r = [](int v){return (v+1)/2;}; // fg -> original graph
	std::vector<int> visited(fg.n+1);
	std::vector<std::vector<int>> cover;
	while(true) {
		std::fill(visited.begin(), visited.end(), 0);
		std::vector<int> path;
		auto dfs = [&fg, &visited, &path](auto dfs, int s)->bool {
			if(s == fg.sink) 
				return true;
			visited[s] = 1;
			for(auto &[u, e]:fg.edge_out[s]) {
				if(visited[u] || e->flow == 0)
					continue;
				if(dfs(dfs, u)) {
					e->flow--;
					path.push_back(u);
					return true;
				}
			}
			return false;
		};
		if(!dfs(dfs, fg.source))
			break;
		path.push_back(fg.source);
		std::vector<int> real_path;
		for(int i=1; i<path.size()-1; i+=2) {
			real_path.push_back(v_r(path[i]));
		}
		std::reverse(real_path.begin(), real_path.end());
		cover.push_back(real_path);
	}

	for(int i=1; i<=fg.n; i++) {
		for(auto &[u, e]:fg.edge_out[i])
			assert(e->flow == 0);
	}
	return cover;
}

// Find augmenting paths 1 by 1 from residual graph
void naive_minflow_solve(Flowgraph<Edge::Minflow> &fg) {
	std::vector<bool> visited(fg.n+1);
	while(true) {
		std::fill(visited.begin(), visited.end(), 0);
		auto dfs = [&fg, &visited](auto dfs, int s)->bool {
			if(s == fg.sink)
				return true;
			visited[s] = 1;
			for(auto &[u, e]:fg.edge_out[s]) {
				if(visited[u] || e->demand >= e->flow)
					continue;
				if(dfs(dfs, u)) {
					e->flow--;
					return true;
				}
			}
			for(auto &[u, e]:fg.edge_in[s]) {
				if(visited[u])
					continue;
				if(dfs(dfs, u)) {
					e->flow++;
					return true;
				}
			}
			return false;
		};
		if(!dfs(dfs, fg.source))
			break;
	}
}

// O(k^2*|V|)?
std::unique_ptr<Flowgraph<Edge::Minflow>> test_reduction(Graph &g) {
	auto fgo = std::make_unique<Flowgraph<Edge::Minflow>>(g.n*2+2, g.n*2+1, g.n*2+2);
	auto v_in = [](int v){return v*2-1;};
	auto v_out = [](int v){return v*2;};
	auto &fg = *fgo;
	std::vector<int> topo;
	std::vector<bool> visited(g.n+1);
	int ans = 0;
	auto dfs = [&g, &ans, &visited, &topo](auto dfs, int s) {
		if(visited[s])
			return;
		visited[s] = 1;
		for(auto &u:g.edge_out[s]) {
			dfs(dfs, u);
		}
		topo.push_back(s);
	};
	for(int i=1; i<=g.n; i++)
		dfs(dfs, i);
	std::reverse(topo.begin(), topo.end());
	std::vector<int> vdx(g.n+1);
	for(int i=0; i<g.n; i++)
		vdx[topo[i]] = i;
	
	std::vector<int> v(g.n+1);
	int PATHS = 0;
	std::vector<int> pver;
	std::vector<std::vector<int>> preaches(g.n+1);
	std::vector<std::vector<int>> preaches2(g.n+1);
	std::vector<int> plen;
	std::vector<int> pend;
	std::vector<std::vector<std::pair<int, Edge::Minflow*>>> from(g.n+1);

	// sparsification to o(k) edge
	std::vector<int> source_flow(g.n+1);
	std::vector<int> node_flow(g.n+1);
	std::vector<int> sink_flow(g.n+1);
	std::vector<int> some_path(g.n+1);
	for(int i=0; i<g.n; i++) {
		int cur = topo[i];
		int bst_len = 0;
		int bst_topo;
		int bst;
		preaches[cur].resize(PATHS);
		from[cur].resize(PATHS, {-1, nullptr});
		// sparsify
		std::vector<int> sparsify(PATHS, -1);
		for(auto &u:g.edge_in[cur]) {
			sparsify[some_path[u]] = std::max(sparsify[some_path[u]], vdx[u]);
		}
		for(auto u:sparsify) {
			if(u == -1)
				continue;
			u = topo[u];
			auto *e = fg.add_edge(v_out(u), v_in(cur));
			for(int j=0; j<preaches2[u].size(); j++) {
				if(preaches[u][preaches2[u][j]] == pver[preaches2[u][j]] && preaches[cur][preaches2[u][j]] == 0) {
					auto curp = preaches2[u][j];
					preaches[cur][preaches2[u][j]] = pver[preaches2[u][j]];
					preaches2[cur].push_back(preaches2[u][j]);
					from[cur][curp] = {u, e};
					if(plen[curp] > bst_len) {
						bst_len = plen[curp];
						bst_topo = vdx[pend[curp]];
						bst = curp;
					} else if(plen[curp] == bst_len && vdx[pend[curp]] < bst_topo) {
						bst_topo = vdx[pend[curp]];
						bst = curp;
					}
				}
			}
		}
		if(bst_len == 0) {
			source_flow[cur]++;
			preaches[cur].push_back(i+1);
			preaches2[cur].push_back(PATHS);
			pver.push_back(i+1);
			plen.push_back(1);
			pend.push_back(cur);
			some_path[cur] = PATHS;
			PATHS++;
		} else {
			plen[bst]++;
			preaches[cur][bst] = i+1;
			pver[bst] = i+1;
			pend[bst] = cur;
			some_path[cur] = bst;
		}
	}
	for(int i=0; i<pend.size(); i++) {
		auto cr = pend[i];
		sink_flow[cr] = 1;
		while(true) {
			node_flow[cr]++;
			if(from[cr].size() < (1+i) || from[cr][i].first == -1)
				break;
			from[cr][i].second->flow++;
			cr = from[cr][i].first;
		}
	}
	for(int i=1; i<=g.n; i++) {
		auto *e = fg.add_edge(fg.source, v_in(i));
		e->flow = source_flow[i] ? 1 : 0;
		auto e2 = fg.add_edge(v_out(i), fg.sink);
		e2->flow = sink_flow[i] ? 1 : 0;
		auto e3 = fg.add_edge(v_in(i), v_out(i));
		e3->flow = node_flow[i];
		e3->demand = 1;
	}
	return fgo;
}

std::unique_ptr<Flowgraph<Edge::Minflow>> greedy_minflow_reduction(Graph &g, std::function<int(int)> node_weight) {
	Flowgraph<Edge::Minflow> tfg = {g.n, 0, 0};
	std::vector<int> topo;
	std::vector<bool> visited(g.n+1);
	auto dfs = [&tfg, &g, &visited, &topo](auto dfs, int s) {
		if(visited[s])
			return;
		visited[s] = 1;
		for(auto &u:g.edge_out[s]) {
			dfs(dfs, u);
			tfg.add_edge(s, u);
		}
		topo.push_back(s);
	};
	for(int i=1; i<=g.n; i++)
		dfs(dfs, i);
	std::reverse(topo.begin(), topo.end());

	struct Node_flow {
		int source;
		int sink;
		int flow;
	};
	std::vector<Node_flow> node_flow(g.n+1);
	std::vector<int> max_len(g.n+1);
	std::vector<std::pair<int, Edge::Minflow*>> from(g.n+1);
	auto &not_covered = visited;
	while(true) {
		std::fill(max_len.begin(), max_len.end(), 0);
		std::pair<int, int> best_node = {0,0};
		for(auto s:topo) {
			if(not_covered[s])
				max_len[s]++;
			if(max_len[s] > best_node.second)
				best_node = {s, max_len[s]};
			if(max_len[s] == 0) { // todo check dense graph?
				continue;
			}
			for(auto &[u,e]:tfg.edge_out[s]) {
				if(max_len[s] > max_len[u]) {
					max_len[u] = max_len[s];
					from[u] = {s, e};
				}
			}
		}
		if(best_node.first == 0)
			break;
		int cur = best_node.first;
		node_flow[cur].sink++;
		while(from[cur].first != 0) {
			not_covered[cur] = 0;
			from[cur].second->flow++;
			node_flow[cur].flow++;
			cur = from[cur].first;
		}
		not_covered[cur] = 0;
		node_flow[cur].flow++;
		node_flow[cur].source++;
	}
	// Reduce to minflow
	int source = g.n*2+1;
	int sink = g.n*2+2;
	auto fgo = std::make_unique<Flowgraph<Edge::Minflow>>(g.n*2+2, source, sink);
	auto v_in = [](int v){return v*2-1;};
	auto v_out = [](int v){return v*2;};
	for(int i=1; i<=g.n; i++) {
		for(auto &[u,e]:tfg.edge_out[i]) {
			auto *e2 = fgo->add_edge(v_out(i), v_in(u));
			e2->flow = e->flow;
		}
		Edge::Minflow *e = fgo->add_edge(v_in(i), v_out(i));
		e->demand = 1;
		e->flow = node_flow[i].flow;
		e = fgo->add_edge(source, v_in(i));
		e->demand = 0;
		e->flow = node_flow[i].source;
		e = fgo->add_edge(v_out(i), sink);
		e->demand = 0;
		e->flow = node_flow[i].sink;
	}
	assert(is_valid_minflow(*fgo));
	return fgo;
}

std::unique_ptr<Flowgraph<Edge::Minflow>> naive_minflow_reduction(Graph &g, std::function<int(int)> node_weight) {
	int source = g.n*2+1;
	int sink = g.n*2+2;
	auto fgo = std::make_unique<Flowgraph<Edge::Minflow>>(g.n*2+2, source, sink);
	auto v_in = [](int v){return v*2-1;};
	auto v_out = [](int v){return v*2;};
	for(int i=1; i<=g.n; i++) {
		for(auto &u:g.edge_out[i]) {
			fgo->add_edge(v_out(i), v_in(u));
		}
		Edge::Minflow *e = fgo->add_edge(v_in(i), v_out(i));
		e->demand = node_weight(i);
		e->flow = node_weight(i);
		e = fgo->add_edge(source, v_in(i));
		e->demand = 0;
		e->flow = node_weight(i);
		e = fgo->add_edge(v_out(i), sink);
		e->demand = 0;
		e->flow = node_weight(i);
	}
	return fgo;
}

bool is_valid_minflow(Flowgraph<Edge::Minflow> &fg) {
	for(int i=1; i<=fg.n; i++) {
		int total_out = 0;
		for(auto &[u,e]:fg.edge_out[i]) {
			if(e->flow < e->demand) {
				std::cout << "Demand not satisfied" << std::endl;
				return false;
			}
			total_out += e->flow;
		}
		int total_in = 0;
		for(auto &[u,e]:fg.edge_in[i])
			total_in += e->flow;
		if(i != fg.sink && i != fg.source && total_in != total_out) {
				std::cout << "Flow conservation not satisfied " << i << " " << total_in << "/" << total_out<< std::endl;
			return false;
		}
	}
	return true;
}

bool is_valid_cover(std::vector<std::vector<int>> &cover, Graph &g) {
	std::vector<int> visited(g.n+1);
	for(auto path:cover) {
		for(auto &u:path)
			visited[u] = 1;
		for(int i=1; i<path.size(); i++)
			if(!g.has_edge(path[i-1], path[i]))
				return false;
	}
	for(int i=1; i<=g.n; i++)
		if(!visited[i])
			return false;
	return true;
}

// int main() {
// 	// Faster input:
// 	std::ios_base::sync_with_stdio(0);
// 	std::cin.tie(0);
	
// 	int n, m;
// 	std::cin >> n >> m;

// 	Graph g(n);
// 	for (int i=0; i<m; i++) {
// 		int a, b;
// 		std::cin >> a >> b;
// 		a++; b++;
// 		g.add_edge(a, b);
// 	}
	
// 	std::vector<int> node_weight(n+1);
// 	for (int i=0; i<n; i++) {
// 		int v, w;
// 		std::cin >> v >> w;
// 		v++;
// 		node_weight[v] = w;
// 	}

// 	Flowgraph<Edge::Minflow> fg = *naive_minflow_reduction(g, [&](int v){
// 			return node_weight[v];
// 	});
// 	naive_minflow_solve(fg);
// 	antichain mwa = maxantichain_from_minflow(fg);

// 	for (int i=0; i<(int)mwa.size(); i++)
// 		std::cout << mwa[i] << " \n"[i+1 == (int)mwa.size()];


// 	return 0;
// }
