#include<bits/stdc++.h>
#include<cuda.h>
using namespace std;

// A structure that stores the flow along a directed edge
struct edge{
	int u, v, c, f;
};

// finds augmenting path using Breadth first search
__global__ void find_augmenting_path(edge *d_edges, int m, int *prefix_deg, 
									 int *in_queue, int *adj, int *vis, int *par, 
									 int *current_flow, int *progress){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m){
		int u = d_edges[id].u, v = d_edges[id].v, c = d_edges[id].c, f = d_edges[id].f;
		// checking if u is in queue and if forward arc uv exists in the residual graph
		if(in_queue[u] && !vis[v] && f < c && atomicCAS(par+v, -1, id) == -1){
			vis[v] = 1;
			current_flow[v] = min(current_flow[u], c - f);
			atomicAdd(progress, 1);
		}
		// checking if v is in queue and if reverse arc vu exists in the residual graph
		if(in_queue[v] && !vis[u] && f && atomicCAS(par+u, -1, id) == -1){
			vis[u] = 1;
			current_flow[u] = min(current_flow[v], f);
			atomicAdd(progress, 1);
		}
	}
}

// adds the vertices to be scanned in next iteration
__global__ void add_to_queue(int n, int *in_queue, int *vis){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n){
		if(vis[id] == 1)
			in_queue[id] = 1;
		else
			in_queue[id] = 0;
	}
}

// removes the vertices that were scanned in the previous iteration
__global__ void remove_from_queue(int n, int *in_queue, int *vis){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n){
		if(in_queue[id] == 1)
			vis[id] = 2;
	}
}

// augments along the path found by find_augmenting_path
__global__ void augment(edge* d_edges, int* par, int t, int flow){
	int cur = t;
	while(cur){
		int idx = par[cur];
		int u = d_edges[idx].u, v = d_edges[idx].v;
		if(cur == u){
			d_edges[idx].f -= flow;
			cur = v;
		}
		else{
			d_edges[idx].f += flow;
			cur = u;
		}
	}
}

int main(int argc, char* argv[]){
	auto clk=clock();
	
	if(argc < 2){
		cout<<"Enter file name"<<endl;
		return 0;
	}

	int n, m, INF = 1000000000, ONE = 1;
	edge *edges, *d_edges;
	int *vis, *par, *current_flow, *in_queue, *progress, *adj, *d_adj, *prefix_deg, *d_prefix_deg;

	ifstream fin(argv[1]);
	fin >> n >> m;
	vector<int> deg(n);
	vector<vector<int>> edge_idx(n);
	edges = new edge[m];
	
	for(int i = 0; i < m; i++){
		fin >> edges[i].u >> edges[i].v >> edges[i].c;
		edges[i].u--;
		edges[i].v--;
		deg[edges[i].u]++;
		deg[edges[i].v]++;
		edge_idx[edges[i].u].push_back(i);
		edge_idx[edges[i].v].push_back(i);
		edges[i].f = 0;
	}	

	prefix_deg = (int*) malloc((n + 1) * sizeof(int));
	prefix_deg[0] = 0;
	for(int i = 1; i <= n; i++){
		prefix_deg[i] = prefix_deg[i-1] + deg[i-1];
	}
	
	adj = (int*) malloc(2 * m * sizeof(int));
	for(int i = 0, c = 0; i < n ; i++){
		for(int j = 0; j < edge_idx[i].size(); j++, c++){
			adj[c] = edge_idx[i][j];
		}
	}

	cudaMalloc(&d_prefix_deg, (n + 1) * sizeof(int));
	cudaMalloc(&d_adj, 2 * m * sizeof(int));
	cudaMalloc(&d_edges, m * sizeof(edge));
	cudaMalloc(&vis, n * sizeof(int));
	cudaMalloc(&par, n * sizeof(int));
	cudaMalloc(&in_queue, n * sizeof(int));
	cudaMalloc(&current_flow, n * sizeof(int));
	cudaMalloc(&progress, sizeof(int));

	cudaMemcpy(d_prefix_deg, prefix_deg, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_adj, adj, 2 * m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, edges, m * sizeof(edge), cudaMemcpyHostToDevice);

	int threads = 1024;
	int m_blocks = ceil((float)m/threads);
	int n_blocks = ceil((float)n/threads);
	int total_flow = 0;

	while(true){

		cudaMemset(vis, 0, n * sizeof(int));
		cudaMemset(par, -1, n * sizeof(int));
		cudaMemset(current_flow, 0, n * sizeof(int));
		cudaMemset(in_queue, 0, n * sizeof(int));

		cudaMemcpy(in_queue, &ONE, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(vis, &ONE, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(current_flow, &INF, sizeof(int), cudaMemcpyHostToDevice);

		int prog, t_reachable, cur_flow;

		// find augmenting path in parallel fashion
		// break when there is no new vertex that is scanned
		do{
			cudaMemset(progress, 0, sizeof(int));
			find_augmenting_path<<<m_blocks,threads>>>(d_edges, m, d_prefix_deg, in_queue, d_adj, 
														vis, par, current_flow, progress);
			cudaMemcpy(&prog, progress, sizeof(int), cudaMemcpyDeviceToHost);
			remove_from_queue<<<n_blocks, threads>>>(n, in_queue, vis);
			add_to_queue<<<n_blocks, threads>>>(n, in_queue, vis);
		
		}while(prog);
		
		cudaMemcpy(&t_reachable, vis + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&cur_flow, current_flow + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

		if(!t_reachable){
			assert(!cur_flow);
			break;
		}

		augment<<<1,1>>>(d_edges, par, n-1 , cur_flow);
		total_flow += cur_flow;
	}

	double t_elapsed = (double)(clock()-clk)/CLOCKS_PER_SEC;
	printf("|V|:%d |E|:%d Flow:%d\nTime:%f\n", n, m, total_flow, t_elapsed);	
}
