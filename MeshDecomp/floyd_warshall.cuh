#ifndef _FLOYD_WARSHALL_
#define _FLOYD_WARSHALL_

#include <iostream>

// CONSTS for CUDA FW
#define BLOCK_SIZE 32
#define MAX_DISTANCE 999999.0

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param weight_matrix: Array of graph with distance between vertex on device, weight_ii = 0, MAX_DISTANCE if no path
 * @param pred_matrix: Array of predecessors for a graph on device, -1 if no path
 * @param distance_matrix: Array of shortest distance between vertex
 * @param n_vertex: Number of vertex
 */
void cudaBlockedFW(void* weight_matrix, void* pred_matrix, void* distance_matrix, size_t n_vertex, size_t* pitch);


#endif