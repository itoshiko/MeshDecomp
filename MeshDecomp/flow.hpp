
#pragma once

#include <cstdio>
#include <cstring>
#include <queue>
#include <stack>
#include <time.h>

using namespace std;
const int INF = 0x3f3f3f3f;

#define EPS 1e-6

struct flowEdgeHLPP {
    int nex, t;
    float v;
    flowEdgeHLPP() {}
    flowEdgeHLPP(int _nex, int _t, float _v)
    {
        nex = _nex;
        t = _t;
        v = _v;
    }
};

struct Edge {
    int from, to;
    float cap, flow;

    Edge(int u, int v, float c, float f) : from(u), to(v), cap(c), flow(f) {}
};

class HLPP
{
public:
    HLPP(int max_vertex, int s, int t) {
        n = max_vertex;
        ex.resize(max_vertex + 1);
        bucket.resize(max_vertex + 1);
        h.resize(max_vertex + 1);
        source = s;
        target = t;
        e.clear();
        e.emplace_back();
        e.emplace_back();
    }

    void add_flow(int f, int t, float v) {
        add_double_path(f, t, v);
        // add_double_path(t, f, v);
    }

private:
    std::vector<flowEdgeHLPP> e;
    std::vector<int> h;

    std::vector<int> height;  // height of nodes
    std::vector<float> ex;
    std::vector<int> gap;
    std::vector<std::stack<int>> bucket;
    int level = 0;
    int cnt = 1;
    int source, target;
    int n;

    void add_double_path(int f, int t, float v) {
        cnt++;
        e.emplace_back(h[f], t, v), h[f] = cnt;
        cnt++;
        e.emplace_back(h[t], f, 0.0), h[t] = cnt;
    }

    int push(int u) {
        bool init = u == source;
        for (int i = h[u]; i; i = e[i].nex) {
            const int& v = e[i].t;
            const float& w = e[i].v;
            if ((abs(w) < EPS) || init == false && height[u] != height[v] + 1)
                continue;
            float k = init ? w : min(w, ex[u]);
            if (v != source && v != target && abs(ex[v]) < EPS) bucket[height[v]].push(v), level = max(level, height[v]);
            ex[u] -= k, ex[v] += k, e[i].v -= k, e[i ^ 1].v += k;  // push
            if (abs(ex[u]) < EPS) return 0;
        }
        return 1;
    }

    void relabel(int u) {
        height[u] = INF;
        for (int i = h[u]; i; i = e[i].nex)
            if (e[i].v > EPS) height[u] = min(height[u], height[e[i].t]);
        if (++height[u] < n) {
            bucket[height[u]].push(u);
            level = max(level, height[u]);
            ++gap[height[u]];
        }
    }

    bool bfs_init() {
        height.assign(n + 1, INF);
        queue<int> q;
        q.push(target), height[target] = 0;
        while (q.size()) {
            int u = q.front();
            q.pop();
            for (int i = h[u]; i; i = e[i].nex) {
                const int& v = e[i].t;
                if ((e[i ^ 1].v > EPS) && height[v] > height[u] + 1)
                {
                    height[v] = height[u] + 1, q.push(v);
                }
            }
        }
        return height[source] != INF;
    }

    int select() {
        while (level > -1 && bucket[level].size() == 0) level--;
        return level == -1 ? 0 : bucket[level].top();
    }

    float calc_flow() {
        if (!bfs_init()) return 0;
        printf("Connectivity: true\n");
        gap.assign(n, 0);
        for (int i = 1; i <= n; i++)
            if (height[i] != INF) gap[height[i]]++;
        height[source] = n;
        push(source);
        int u;
        while ((u = select())) {
            bucket[level].pop();
            if (push(u)) {
                if (!--gap[height[u]])
                    for (int i = 1; i <= n; i++)
                        if (i != source && i != target && height[i] > height[u] && height[i] < n + 1)
                            height[i] = n + 1;
                relabel(u);
            }
        }
        // for (int i = 0; i < t; i++) printf("%f  ", ex[t]);
        printf("Max flow: %f\n", ex[target]);
        return ex[target];
    }
};


class Dinic 
{
public:
    Dinic(int max_vertex) {
        n = max_vertex;
        m = 0;
        s = 0;
        t = 0;
        G.resize(max_vertex + 1);
        d.resize(max_vertex + 1);
        for (int i = 0; i < n; i++) G[i].clear();
        edges.clear();
    }

    void AddEdge(int from, int to, float cap) {
        edges.emplace_back(from, to, cap, 0.0);
        edges.emplace_back(to, from, 0.0, 0.0);
        m = edges.size();
        G[from].push_back(m - 2);
        G[to].push_back(m - 1);
    }

    void cut_classify(int s, int t, std::vector<int>& id) {
        if (edges.empty())
        {
            id.clear();
            printf("Empty graph!\n");
            return;
        }
        Maxflow(s, t);
        id.clear();
        visited.assign(n + 1, 0);
        queue<int> Q;
        Q.push(s);
        d[s] = 0;
        visited[s] = 1;
        while (!Q.empty()) {
            int x = Q.front();
            Q.pop();
            for (int i = 0; i < G[x].size(); i++) {
                Edge& e = edges[G[x][i]];
                if (!visited[e.to]) {
                    if (e.cap > e.flow)
                    {
                        visited[e.to] = 1;
                        d[e.to] = d[x] + 1;
                        Q.push(e.to);
                        id.push_back(e.to);
                    }
                    else
                    {
                        // id.push_back(e.to);
                    }
                }
            }
        }
    }

private:
    int n, m, s, t;
    vector<Edge> edges;
    vector<vector<int>> G;
    vector<int> d, cur;
    vector<bool> visited;

    bool BFS() {
        visited.assign(n + 1, 0);
        queue<int> Q;
        Q.push(s);
        d[s] = 0;
        visited[s] = 1;
        while (!Q.empty()) {
            int x = Q.front();
            Q.pop();
            for (int i = 0; i < G[x].size(); i++) {
                Edge& e = edges[G[x][i]];
                if (!visited[e.to] && e.cap > e.flow) {
                    visited[e.to] = 1;
                    d[e.to] = d[x] + 1;
                    Q.push(e.to);
                }
            }
        }
        return visited[t];
    }

    float DFS(int x, float a) {
        if (x == t || a == 0) return a;
        float flow = 0.0, f;
        for (int& i = cur[x]; i < G[x].size(); i++) {
            Edge& e = edges[G[x][i]];
            if (d[x] + 1 == d[e.to] && (f = DFS(e.to, min(a, e.cap - e.flow))) > 0) {
                e.flow += f;
                edges[G[x][i] ^ 1].flow -= f;
                flow += f;
                a -= f;
                if (abs(a) < 1e-6) break;
            }
        }
        return flow;
    }

    float Maxflow(int s, int t) {
        this->s = s;
        this->t = t;
        float flow = 0;
        while (BFS()) {
            cur.assign(n + 1, 0);
            flow += DFS(s, INF);
        }
        printf("Max flow: %f\n", flow);
        return flow;
    }
};


//int main_hlpp() {
//    FILE* fin;
//    fin = fopen("flow1.txt", "r");
//    int n, m, s, t;
//    fscanf(fin, "%d%d%d%d", &n, &m, &s, &t);
//    HLPP hlpp = HLPP(n, s, t);
//    for (int i = 1; i <= m; i++) {
//        int u;
//        int v;
//        float w;
//        fscanf(fin, "%d%d%f", &u, &v, &w);
//        hlpp.add_flow(u, v, w);
//    }
//    std::vector<int> class_0;
//    clock_t start = clock();
//    // hlpp.cut_classify(class_0);
//    for (auto i : class_0)
//        printf("%d\n", i);
//    printf("%f\n", double(clock() - start));
//    return 0;
//}
//
//
//int main_dinic() {
//    FILE* fin;
//    fin = fopen("flow.txt", "r");
//    int n, m, s, t;
//    fscanf(fin, "%d%d%d%d", &n, &m, &s, &t);
//    Dinic di = Dinic(n);
//    for (int i = 1; i <= m; i++) {
//        int u;
//        int v;
//        float w;
//        fscanf(fin, "%d%d%f", &u, &v, &w);
//        di.AddEdge(u, v, w);
//        di.AddEdge(v, u, w);
//    }
//    std::vector<int> class_0;
//    clock_t start = clock();
//    di.cut_classify(s, t, class_0);
//    printf("%f\n", double(clock() - start));
//    for (auto i : class_0)
//        printf("%d\n", i);
//    return 0;
//}