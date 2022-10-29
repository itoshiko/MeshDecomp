#include "push_relabel.h"

#define EPS 1e-6

FuzzyGraph::FuzzyGraph(int n_vertex)
{
    n = n_vertex;
    source = n_vertex;
    sink = n_vertex - 1;
    next_edge.resize(2);
    capacity.resize(2);
    to.resize(2);
    first.resize((size_t)n_vertex + 5);
    flow.assign((size_t)n_vertex + 5, 0.);
    cnt.assign((size_t)n_vertex + 5, 0);
    height.assign((size_t)n_vertex + 5, 0);
}

void FuzzyGraph::add_edge(int u, int v, float cap)
{
    edge_cnt++;
    next_edge.push_back(first[u]);
    first[u] = edge_cnt;
    to.push_back(v);
    capacity.push_back(cap);
    edge_cnt++;
    next_edge.push_back(first[v]);
    first[v] = edge_cnt;
    to.push_back(u);
    capacity.push_back(0.);
}

float FuzzyGraph::push(int u, int v, int e)
{
    float delta = fminf(capacity[e], flow[u]);  // 'some water'
    capacity[e] -= delta;
    capacity[e ^ 1] += delta;
    flow[u] -= delta;
    flow[v] += delta;
    return delta;
}

float FuzzyGraph::HLPP_flow()
{
	// initialisation. push source
	height[source] = n;
	// Set the usable water in source infinitive
	flow[source] = 999999.;
	q.push(node(source, height[source]));
	// Do the push operations.
	while (!q.empty())
	{
		// Get the point. Pop the top.
		int u = q.top().i;
		q.pop();
		// Make sure there's remaining water.
		if (flow[u] < EPS)
			continue;
		for (int e = first[u], v = to[e]; e; e = next_edge[e], v = to[e])
		{
			// Make sure the arc is valid. 
			if ((u == source || height[u] == height[v] + 1)
				// Push some water and make sure the pushed isn't null.
				&& (push(u, v, e) > EPS)
				// You can't push the s/t into the queue, or you'll err.
				&& v != source && v != sink)
				// Push the next one into the queue.
				q.push(node(v, height[v]));
		}
		// If there's remaining things in the point which means there's a lake,
		if (u != source && u != sink && (flow[u] > EPS))
		{
			// If the height is wrong
			if (!(--cnt[height[u]]))
			{
				// make all the inappropriate points' heights higher than s which are intouchable.
				for (int i = 1; i <= n; ++i)
				{
					if (i != source && i != sink && height[i] > height[u] && height[i] <= n) height[i] = n + 1;
				}
			}
			++cnt[++height[u]];  // Highten the point.
			q.push(node(u, height[u]));  // Push it into the queue again.
		}
	}
	printf("Max flow: %f\n", flow[sink]);
	return flow[sink];//The remaining water in the t is the maximum of flow.
}

void FuzzyGraph::min_cut_classify(int* type, int code1, int code2, size_t cnt)
{
    bool* visited = new bool[n + 1];
    memset(visited, 0, n * sizeof(bool));

    std::queue<int> tmp_q;
    tmp_q.push(source);
	visited[source] = true;
    
	int v, c;
    while (!tmp_q.empty())
    {
		// Get the point. Pop the top.
		v = tmp_q.front();
		tmp_q.pop();
		for (int e = first[v], v = to[e]; e; e = next_edge[e], v = to[e])
		{
			// capacity not 0
			if (!visited[v] && (capacity[e] > EPS))
			{
				if (v <= n - 2 && type[cnt + v - 1] >= 0)
				{
					// printf("%d 0\n", v - 1);
					type[v - 1] = code1;
					type[cnt + v - 1] = -1;
				}
					
				visited[v] = true;
				tmp_q.push(v);
			}
		}
    }

	// Do again
	memset(visited, 0, n * sizeof(bool));
	tmp_q.push(source);
	visited[source] = true;
	while (!tmp_q.empty())
	{
		v = tmp_q.front();
		tmp_q.pop();
		for (int e = first[v], v = to[e]; e; e = next_edge[e], v = to[e])
		{
			if (!visited[v])
			{
				// fuzzy vert not visited in first BFS
				if (v <= n - 2 && type[cnt + v - 1] >= 0)
				{
					// printf("%d 1\n", v - 1);
					type[v - 1] = code2;
				}
					
				visited[v] = true;
				tmp_q.push(v);
			}
		}
	}
}

int test_hlpp()
{
	Mesh mesh("d:/course_proj/meshdecomp/asset/smpl_mesh.ply");
	mesh.preProcess();
	// int n = mesh.num_faces;
	int n = 1;
	float* weight = new float[n * n];
	// cudaMemcpy(weight, mesh.graph_weight_matrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);


	FuzzyGraph g(n + 2);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (weight[i * n + j] < 99999.)
			{
				g.add_edge(i + 1, j + 1, weight[i * n + j]);
				g.add_edge(j + 1, i + 1, weight[i * n + j]);
			}
		}
	}
	g.add_edge(n + 2, 3327, 9999.);
	g.add_edge(12400, n + 1, 9999.);
	clock_t t1 = clock();
	printf("%f\n", g.HLPP_flow());
	printf("%f\n", (float)(clock() - t1));
	return 0;
}
