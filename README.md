
**Parallel implementation of standard flow algorithms using CUDA**

Input graphs must in this format (1-indexed):
```
<num_vertices> <num_edges>
<u1> <v1> <capacity1>
<u2> <v2> <capacity2>
...
<um> <vm> <capacitym>
```

You can use `create_graph` as mentioned below.

**USAGE**

Make sure you have CUDA installed in your machine.

Use Makefile to build all binaries 
```
make
```

You can generate a random graph with a given specification using `create_graph`. Usage is as follows:
```
./create_graph <num_vertices> <num_edges> <maximum capacity> > <graph_filename>
```

The following binaries are generated:
```
ford_fulkerson_cpu
ford_fulkerson_gpu
edmond_karp_cpu
edmond_karp_gpu
push_relabel_gpu
```
For running `push_relabel_gpu`, use:
```
./push_relabel_gpu <graph_filename> 0 # Without global relabelling heuristic
./push relabel_gpu <graph_filename> 1 # With global relabelling heuristic
```
For other binaries, use:
```
./<binary> <graph_filename>
```


