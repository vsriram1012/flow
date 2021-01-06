#include<bits/stdc++.h>
using namespace std;
#define ii pair<int,int>
#define vi vector<int>
#define vvi vector<vi>
#define pb push_back
#define mp make_pair
#define INF 1000000000
#define F first
#define S second
struct flow
{
    int n,isDir;
    vvi g,rg;
    int s,t;
    flow(int _n,int _isDir=true)   :n(_n),isDir(_isDir){
        g.resize(n);
        rg.assign(n,vi(n,0));
    }
    void addEdge(int u,int v,int w=1)
    {
        g[u].pb(v);
        g[v].pb(u);
        rg[u][v]=w;
        if(!isDir)
            rg[v][u]=w;
    }
    int sendFlow() 
    {
        vi p(n,-1);
        queue<ii> q;
        q.push(mp(s,INF));
        int flow=0;
        while(!q.empty())
        {
            ii tmp=q.front();
            q.pop();
            int u=tmp.F,f=tmp.S;
            if(u==t)    
            {
                flow=f;
                break;
            }
            for(int v:g[u])
            {
                if(p[v]==-1 && rg[u][v])
                {
                    p[v]=u;
                    int df=min(f,rg[u][v]);
                    q.push(mp(v,df));
                }
            }
        }
        if(!flow)   return 0;
        int cur=t;
        while(cur!=s)
        {
            int pcur=p[cur];
            rg[cur][pcur]+=flow;
            rg[pcur][cur]-=flow;
            cur=pcur;
        }
        return flow;
    }
    int maxflow(int _s,int _t)
    {
        s=_s,t=_t;
        int res=0;
        int flow;
        while(flow=sendFlow())
            res+=flow;
        return res;
    }

};
int main(int argc, char* argv[]){
	if(argc < 2){
		cout<<"Enter file name"<<endl;
		return 0;
	}
	int n, m;
	ifstream fin(argv[1]);
	fin >> n >> m;
	flow graf(n);
	for(int i=0;i<m;i++){
		int u,v,c;
		fin>>u>>v>>c;
		u--;v--;
		graf.addEdge(u,v,c);
	}
	auto clk=clock();
	int total_flow = graf.maxflow(0,n-1);
	double t_elapsed = (double)(clock()-clk)/CLOCKS_PER_SEC;
	printf("|V|:%d |E|:%d Flow:%d Time:%f\n", n, m, total_flow, t_elapsed);	
}