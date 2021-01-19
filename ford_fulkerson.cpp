#include<bits/stdc++.h>
using namespace std;
#define INF 1000000000

// A structure that stores the flow along a directed edge
struct edge{
	int u, v, c, f;
};

// This method uses dfs to find augmenting path
struct FordFulkerson{
	int n;
	vector<edge> edges;
	vector<int> vis, par, val;
	vector<vector<int>> adj;

	FordFulkerson(int _n){
		n = _n;
		adj.resize(n);
	}

	void addEdge(int u, int v, int c){
		adj[u].push_back(edges.size());
		adj[v].push_back(edges.size());
		edge e = {u, v, c, 0};
		edges.push_back(e);
	}

	// finds augmenting path using DFS
	void find_augmenting_path(int u){
		vis[u] = 1;
		for(int id: adj[u]){
			int a = edges[id].u, b = edges[id].v, c = edges[id].c, f = edges[id].f;
			// checking if forward edge uv is present in residual graph
			if(a == u && f < c && !vis[b]){
				val[b] = min(val[a], c - f);
				par[b] = id;
				find_augmenting_path(b);
			}

			// checking if reverse edge vu is present in residual graph
			if(b == u && f > 0 && !vis[a]){
				val[a] = min(val[b],f);
				par[a] = id;
				find_augmenting_path(a);
			}
		}
	}

	// augments along the path found by find_augmenting_path
	void augment(int s, int t, int flow){
		int cur = t;
		while(cur != s){
			int id = par[cur];
			int a = edges[id].u, b = edges[id].v;
			if(b == cur){
				edges[id].f += flow;
				cur = a;
			}
			else{
				edges[id].f -= flow;
				cur = b;
			}
		}
	}

	// computes maxflow from s to t
	int maxflow(int s,int t){
		int total_flow = 0;
		while(true){
			vis.assign(n, 0);
			par.assign(n, -1);
			val.assign(n, 0);
			val[s] = INF;
			find_augmenting_path(s);
			// checks if end point t is reachable from s, breaks if false
			if(!vis[t])	break;
			total_flow += val[t];
			augment(s, t, val[t]);
		}
		return total_flow;
	}
};

int main(int argc, char* argv[]){
	auto clk = clock();
	if(argc < 2){
		cout<<"Enter input graph (filename)"<<endl;
		return 0;
	}

	int n, m;
	ifstream fin(argv[1]);
	fin >> n >> m;

	FordFulkerson graf(n);

	for(int i=0;i<m;i++){
		int u, v, c;
		fin >> u >> v >> c;
		u--; v--;
		graf.addEdge(u, v, c);
	}

	//computes total flow
	int total_flow = graf.maxflow(0,n-1);

	double t_elapsed = (double)(clock()-clk)/CLOCKS_PER_SEC;
	printf("|V|:%d |E|:%d Flow:%d\nTime:%f\n", n, m, total_flow, t_elapsed);	
	return 0;
}