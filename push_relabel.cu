#include<bits/stdc++.h>
#include<cuda.h>
using namespace std;

struct edge{
	int u, v, c, f;
};

// pushes flow along the edges adjacent to a vertex, concurrently for all vertices
__global__ void push(int n, int* excess, int* excess_inc, int* prefix_deg, int* adj, int* height, int* new_height, 
						edge* d_edges){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// push flow only for intermediate vertices (non-terminals)
	if(id > 0 && id < n - 1){	

		new_height[id] = 3 * n;	
		
		for(int i = prefix_deg[id]; i < prefix_deg[id + 1] && excess[id]; i++){
			int idx = adj[i];
			int u = d_edges[idx].u, v = d_edges[idx].v, c = d_edges[idx].c, f = d_edges[idx].f;
			
			// pushes flow along forward edge
			if(u == id && f < c && height[u] > height[v]){
				int push_flow = min(c - f, excess[u]);
				atomicAdd(excess_inc + v, push_flow);
				atomicAdd(&(d_edges[idx].f), push_flow);
				excess[u] -= push_flow;
			}

			// pushes flow along reverse edge
			if(v == id && f && height[v] > height[u]){
				int push_flow = min(f, excess[v]);
				atomicAdd(excess_inc + u, push_flow);
				atomicAdd(&(d_edges[idx].f), -push_flow);
				excess[v] -= push_flow;
			}
		}
	}
}

// computes labels (out of place)
__global__ void compute_label(int n, int m, int* excess, int* excess_inc, int* height, int* new_height, edge* d_edges){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m){
		int u = d_edges[id].u, v = d_edges[id].v, c = d_edges[id].c, f = d_edges[id].f;
		if(u > 0 && u < n - 1 && (excess[u] || excess_inc[u]) && f < c)
			atomicMin(new_height + u, height[v] + 1);
		if(v > 0 && v < n - 1 && (excess[v] || excess_inc[v]) && f)
			atomicMin(new_height + v, height[u] + 1);
	}
}

// applies the labels found in computer_label and updates excess of each vertex
__global__ void relabel(int n, int* excess, int* excess_inc, int* height, int* new_height, int* is_excess){	

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id > 0 && id < n - 1){
		if(new_height[id] != 3 * n)
			height[id] = new_height[id];
		excess[id] += excess_inc[id];
		excess_inc[id] = 0;
		if(excess[id])
			atomicAdd(is_excess, 1);
	}
}

// computes the flow out of source 
__global__ void compute_maxflow(int m, int* total_flow, edge* d_edges)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m){
		int u = d_edges[id].u, v = d_edges[id].v, f = d_edges[id].f;
		if(!u)	atomicAdd(total_flow, f);
		if(!v)	atomicAdd(total_flow, -f);
	}
}

// global relabeling heuristic - performs BFS from sink to find lower bound on labels
__global__ void global_label(int m, int cur_wave, int* height, int* wave, int* progress, edge* d_edges)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m){
		int u = d_edges[id].u, v = d_edges[id].v, c = d_edges[id].c, f = d_edges[id].f;
		if(wave[v] == cur_wave && f < c && atomicCAS(wave + u, -1, cur_wave + 1) == -1){
			height[u] = height[v] + 1;
			atomicAdd(progress, 1);
		}
		if(wave[u] == cur_wave && f && atomicCAS(wave + v, -1, cur_wave + 1) == -1){
			height[v]=height[u] + 1;
			atomicAdd(progress, 1);
		}

	}
}

int main(int argc, char* argv[]){

	auto clk=clock();

	if(argc < 3){
		cout<<"Enter file name and global relabel heuristic flag (0 or 1)"<<endl;
		return 0;
	}

	int n, m;
	edge *edges, *d_edges;
	int *excess, *d_excess, *prefix_deg, *d_prefix_deg, *adj, *d_adj, *d_excess_inc, *d_is_excess, *height, *new_height, 
		*d_total_flow, *progress, *wave;

	ifstream fin(argv[1]);
	int global_relabel_flag = atoi(argv[2]);
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

	int* reset_height = new int[n];
	for(int i = 0; i < n;i++)
		reset_height[i] = n;

	excess = (int*) malloc(n * sizeof(int));
	memset(excess, 0, n * sizeof(int));
	for(int i: edge_idx[0]){
		int u = edges[i].u, v = edges[i].v, c = edges[i].c;
		if(!u){
			edges[i].f = edges[i].c;
			excess[v] += c; 
		}
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

	cudaMalloc(&d_excess, n * sizeof(int));
	cudaMalloc(&d_excess_inc, n * sizeof(int));
	cudaMalloc(&progress, sizeof(int));
	cudaMalloc(&wave, n * sizeof(int));
	cudaMalloc(&d_prefix_deg, (n + 1) * sizeof(int));
	cudaMalloc(&d_adj, 2 * m * sizeof(int));
	cudaMalloc(&height, n * sizeof(int));
	cudaMalloc(&new_height, n * sizeof(int));
	cudaMalloc(&d_edges, m * sizeof(edge));
	cudaMalloc(&d_is_excess, sizeof(int));
	
	cudaMemset(height, 0, n * sizeof(int));
	cudaMemset(d_excess_inc, 0, n * sizeof(int));

	cudaMemcpy(d_excess, excess, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prefix_deg, prefix_deg, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_adj, adj, 2 * m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(height, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, edges, m * sizeof(edge), cudaMemcpyHostToDevice);

	int threads = 1024;
	int m_blocks = ceil((float)m/threads);
	int n_blocks = ceil((float)n/threads);
	int total_flow = 0;
	int is_excess;
	int iter = 0,prog;

	// loop to push flow along edges
	// if there is no excess vertex, the loop breaks
	do{
		iter++;
		push<<<n_blocks, threads>>>(n, d_excess, d_excess_inc, d_prefix_deg, d_adj, height, new_height, d_edges);
		compute_label<<<m_blocks, threads>>>(n, m, d_excess, d_excess_inc, height, new_height, d_edges);
		cudaMemset(d_is_excess, 0, sizeof(int));
		relabel<<<n_blocks, threads>>>(n, d_excess, d_excess_inc, height, new_height, d_is_excess);
		cudaMemcpy(&is_excess, d_is_excess, sizeof(int), cudaMemcpyDeviceToHost);
		
		// applies global relabeling every n iterations
		if(global_relabel_flag && iter % n == 0){	// perform global relabeling heuristic
			cudaMemset(wave, -1, n * sizeof(int));
			cudaMemset(wave + (n - 1) , 0, sizeof(int));
			int cur_wave = 0;
			do{
				cudaMemset(progress, 0, sizeof(int));
				global_label<<<m_blocks, threads>>>(m, cur_wave++, height, wave, progress, d_edges);
				cudaMemcpy(&prog, progress, sizeof(int), cudaMemcpyDeviceToHost);
			}while(prog);
		}

	}while(is_excess);

	cudaMalloc(&d_total_flow, sizeof(int));
	cudaMemset(d_total_flow, 0, sizeof(int));
	compute_maxflow<<<m_blocks, threads>>>(m, d_total_flow, d_edges);
	cudaMemcpy(&total_flow, d_total_flow, sizeof(int), cudaMemcpyDeviceToHost);

	double t_elapsed = (double)(clock()-clk)/CLOCKS_PER_SEC;
	printf("|V|:%d |E|:%d Flow:%d\nTime:%f\n", n, m, total_flow, t_elapsed);	
}