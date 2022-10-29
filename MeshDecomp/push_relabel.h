#pragma once
#include <vector>
#include <queue>
#include <time.h>
#include <iostream>
#include "Mesh.h"

struct node
{
	int i, h;  // The point & the height of the point
	node(int i = 0, int h = 0) :i(i), h(h) {}
	bool operator < (const node& a) const { return h < a.h; }  // Higher one has the priority. Remember the format, don't forget the 'const' and the '&'.
};

struct FuzzyGraph
{
	int n;  // The number of points
	int source, sink;
	std::vector<int> first;  // first edge of a point
	std::vector<int> next_edge;  // next edge of a edge
	std::vector<float> capacity;  // capacity of a edge
	std::vector<int> to;  // the ending to a edge
	int edge_cnt = 1;  // cnt of a edge
	std::vector<float> flow;  // the remaining water in a point
	std::vector<int> cnt;  // count of heights
	std::vector<int> height;  // heights

	std::vector<int> fuzzy_vert;
	std::priority_queue<node> q;  //the priority queue of active nodes

	FuzzyGraph() { n = sink = source = 0; }
	FuzzyGraph(int n_vertex);
	void add_edge(int u, int v, float cap);
	float push(int u, int v, int e);  // Push some water from u to v through e.
	float HLPP_flow();
	void min_cut_classify(int* type, int code, int code2, size_t cnt);
	bool is_graph_empty() { return capacity.size() <= 2; }
};

int test_hlpp();
