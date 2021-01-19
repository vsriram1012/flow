#include<bits/stdc++.h>
#include<cuda.h>
using namespace std;

// A structure that stores the flow along a directed edge
struct edge{
	int u, v, c, f;
};

// finds augmenting path in non-deterministic fashion
__global__ void find_augmenting_path(edge *d_edges, int m, int *vis, int *par, 
									 int *current_flow, int *progress){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < m){
		int u = d_edges[id].u, v = d_edges[id].v, c = d_edges[id].c, f = d_edges[id].f;
		// checking if forward edge uv exists in residual graph
		if(vis[u] && !vis[v] && f < c && atomicCAS(par+v, -1, id) == -1){
			vis[v] = 1;
			current_flow[v] = min(current_flow[u], c - f);
			atomicAdd(progress, 1);
		}
		// checking if reverse edge vu exists in residual graph
		if(vis[v] && !vis[u] && f > 0 && atomicCAS(par+u, -1, id) == -1){
			vis[u] = 1;
			current_flow[u] = min(current_flow[v], f);
			atomicAdd(progress, 1);
		}
	}

}

// augemnts along path found by find_augmenting_path
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

	int n, m, INF = 1000000000;
	edge *edges, *d_edges;
	int *vis, *par, *progress, *current_flow;	

	ifstream fin(argv[1]);
	fin >> n >> m;

	edges = new edge[m];
	for(int i = 0; i < m; i++){
		fin >> edges[i].u >> edges[i].v >> edges[i].c;
		edges[i].u--;
		edges[i].v--;
		edges[i].f = 0;
	}	
	
	cudaMalloc(&d_edges, m * sizeof(edge));
	cudaMalloc(&vis, n * sizeof(int));
	cudaMalloc(&par, n * sizeof(int));
	cudaMalloc(&current_flow, n * sizeof(int));
	cudaMalloc(&progress, sizeof(int));

	cudaMemcpy(d_edges, edges, m*sizeof(edge), cudaMemcpyHostToDevice);

	int threads = 1024;
	int blocks = ceil((float)m/threads);
	int total_flow = 0;

	while(true){

		cudaMemset(vis, 0, n * sizeof(int));
		cudaMemset(par, -1, n * sizeof(int));
		cudaMemset(current_flow, 0, n * sizeof(int));
		cudaMemset(vis, 1, sizeof(int));

		cudaMemcpy(current_flow, &INF, sizeof(int), cudaMemcpyHostToDevice);

		int prog, t_reachable, cur_flow;
		
		// this loop performs search for augmenting path in parallel fashion
		// loop breaks when there is no new vertex that is reached in the last iteration
		do{
			cudaMemset(progress, 0, sizeof(int));
			find_augmenting_path<<<blocks,threads>>>(d_edges, m, vis, par, current_flow, progress);
			cudaMemcpy(&prog, progress, sizeof(int), cudaMemcpyDeviceToHost);
		
		}while(prog);

		cudaMemcpy(&t_reachable, vis + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&cur_flow, current_flow + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		
		if(!t_reachable){
			assert(!cur_flow);
			break;
		}
		
		// has to be done serially
		augment<<<1,1>>>(d_edges, par, n-1 , cur_flow);
		
		total_flow += cur_flow;
	}

	double t_elapsed = (double)(clock()-clk)/CLOCKS_PER_SEC;
	printf("|V|:%d |E|:%d Flow:%d\nTime:%f\n", n, m, total_flow, t_elapsed);	
}