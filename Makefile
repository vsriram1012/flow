all:
	g++ create_graph.cpp -o create_graph
	g++ ford_fulkerson.cpp -o ford_fulkerson_cpu
	g++ edmond_karp.cpp -o edmond_karp_cpu
	nvcc ford_fulkerson.cu -o ford_fulkerson_gpu
	nvcc edmond_karp.cu -o edmond_karp_gpu
	nvcc push_relabel.cu -o push_relabel_gpu

clean:
	rm -f create_graph ford_fulkerson_cpu edmond_karp_cpu ford_fulkerson_gpu edmond_karp_gpu push_relabel_gpu