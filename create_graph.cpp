#include<bits/stdc++.h>
using namespace std;

mt19937_64 rang(chrono::high_resolution_clock::now().time_since_epoch().count());
int rng(int lim) {uniform_int_distribution<int> uid(0,lim-1);return uid(rang);}

// creates a random graph with |V| vertices and |E| edges
int main(signed argc, char* argv[]){
	if(argc<3){
        cout<<"Enter |V| and |E| and max-capacity (default to 1000)"<<endl;
        return 0;
    }
    
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int C = 1000;

    if(argc > 3)
        C = atoi(argv[3]);
	
    cout << n << ' ' << m << endl; 

    set<pair<int,int>> edges;

    for(int i = 0;i < m; i++){

        int u = rng(n), v = rng(n), c = rng(C)+1;
        while(u == v){
            u = rng(n), v= rng(n);
        }
        cout << u + 1 << ' ' << v + 1 << ' ' << c << endl;
    }
}