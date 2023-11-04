#include "naive.h"
#include "graph.h"
#include "antichain.h"
#include <vector>
#include <stack>


// TODO fix - ei toimi
antichain maxantichain_from_mpc(Graph &g, path_cover &mpc) {
	// mpc -> minflow
	// topo order . . => linear minflow graph 
	// reverse all 1way flow topo order
	int source = g.n*2+1;
	int sink = g.n*2+2;
	auto fgo = Flowgraph<Edge::Minflow>(g.n*2+2, source, sink);
	auto v_in = [](int v){return v*2-1;};
	auto v_out = [](int v){return v*2;};
	std::vector<int> visited(g.n+1);
	std::vector<int> ree(g.n+1);
	std::vector<std::vector<std::pair<int, std::stack<int>>>> flowcount(g.n+1);
	// todo stack copy
	for(auto &path:mpc) {
		flowcount[path[0]].push_back({0, {}});
		for(auto it=path.rbegin(); it!=path.rend()-1; it++) {
			flowcount[path[0]][flowcount[path[0]].size()-1].second.push(*it);
		}
	}
	for(int i=1; i<=g.n; i++) {
		Edge::Minflow *e = fgo.add_edge(source, v_in(i));
		e->demand = 0;
		e->flow = flowcount[i].size();
	}
	auto dfs = [&ree, &v_in, &v_out, &fgo, &g, &flowcount, &visited](auto dfs, int s) {
		if(visited[s])
			return;
		visited[s] = 1;
		for(auto &u:g.edge_in[s]) {
			dfs(dfs, u);
		}
		auto *e = fgo.add_edge(v_in(s), v_out(s));
		e->demand = 1;
		e->flow = flowcount[s].size();
		assert(e->flow>0);
		int pv = 0;
		int cnt = 0;
		for(auto &u:flowcount[s]) {
			if(cnt && pv && pv != u.first) {
				auto *e = fgo.add_edge(v_out(pv), v_in(s));
				e->demand = 0;
				e->flow = cnt;
				cnt = 0;
			}
			cnt++;
			pv = u.first;
			if(u.second.size() >= 1) {
				flowcount[u.second.top()].push_back({s, u.second});
				assert(visited[u.second.top()]<2);
				flowcount[u.second.top()][flowcount[u.second.top()].size()-1].second.pop();
			} else {
				ree[s]++;
			}
		}
		if(cnt && pv) {
			auto *e = fgo.add_edge(v_out(pv), v_in(s));
			e->demand = 0;
			e->flow = cnt;
			cnt = 0;
		}
		for(auto &u:g.edge_out[s])
			if(flowcount[u].size() == 0)
				fgo.add_edge(v_out(s), v_in(u));
		visited[s] = 2;
	};
	for(int i=1; i<=g.n; i++)
		dfs(dfs, i);
	for(int i=1; i<=g.n; i++) {
		auto *e = fgo.add_edge(v_out(i), sink);
		e->flow = ree[i];
	}
	assert(is_valid_minflow(fgo));
	return maxantichain_from_minflow(fgo);
}

antichain maxantichain_from_minflow(Flowgraph<Edge::Minflow> &mf) {
	std::vector<int> visited(mf.n+1);
	auto v_r = [](int v){return (v+1)/2;}; // fg -> original graph
	antichain mac;
	auto dfs = [&mac, &v_r, &mf, &visited](auto &dfs, int s) {
		if(visited[s])
			return;
		assert(s != mf.sink);
		visited[s] = 1;
		for(auto &[u,e] : mf.edge_out[s]) {
			if(e->flow > e->demand) {
				dfs(dfs, u);
			}
		}
		for(auto &[u, e] : mf.edge_in[s]) {
			dfs(dfs, u);
		}
	};
	dfs(dfs, mf.source);
	auto dfs2 = [&mac, &v_r, &mf, &visited](auto &dfs, int s) {
		if(visited[s] != 1)
			return;
		visited[s] = 2;
		for(auto &[u,e] : mf.edge_out[s]) {
			if(e->flow > e->demand) {
				dfs(dfs, u);
			}
			if(e->flow == e->demand && e->demand >= 1 && !visited[u]) {
				mac.push_back(v_r(s));
				visited[u] = 3; 
			}

		}
		for(auto &[u, e] : mf.edge_in[s]) {
			dfs(dfs, u);
		}
	};
	dfs2(dfs2, mf.source);
	return mac;
}

bool is_antichain(antichain &ac, Graph &g) {
	std::vector<bool> visited(g.n+1), antichain(g.n+1);
	for(auto u:ac)
		antichain[u] = 1;
	auto dfs = [&g, &visited, &antichain](auto &dfs, int s)->bool {
		if(visited[s])
			return false;
		if(antichain[s]) {
			return true;
		}
		visited[s] = 1;
		for(auto u:g.edge_out[s]) {
			if(dfs(dfs, u))
				return true;
		}
		return false;
	};
	for(auto u:ac) {
		antichain[u] = 0;
		if(dfs(dfs, u))
			return false;
		antichain[u] = 1;
		for(int i=1; i<=g.n; i++)
			visited[i] = 0;
	}
	return true;
}
