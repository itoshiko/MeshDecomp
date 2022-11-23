#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "floyd_warshall.cuh"
#include "cu_common_util.cuh"

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Dependent phase 1
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const size_t nvertex, float* const graph, int* const pred) {
    __shared__ float cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePred[BLOCK_SIZE][BLOCK_SIZE];

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = BLOCK_SIZE * blockId + idy;
    const int v2 = BLOCK_SIZE * blockId + idx;

    int newPred;
    float newPath;

    const int cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraph[idy][idx] = graph[cellId];
        cachePred[idy][idx] = pred[cellId];
        newPred = cachePred[idy][idx];
    }
    else {
        cacheGraph[idy][idx] = MAX_DISTANCE;
        cachePred[idy][idx] = -1;
    }

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

#pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        newPath = cacheGraph[idy][u] + cacheGraph[u][idx];

        // Synchronize before calculate new value
        __syncthreads();
        if (newPath < cacheGraph[idy][idx]) {
            cacheGraph[idy][idx] = newPath;
            newPred = cachePred[u][idx];
        }

        // Synchronize to make sure that all value are current
        __syncthreads();
        cachePred[idy][idx] = newPred;
    }

    if (v1 < nvertex && v2 < nvertex)
    {
        graph[cellId] = cacheGraph[idy][idx];
        pred[cellId] = cachePred[idy][idx];
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Partial dependent phase 2
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, const size_t nvertex, float* const graph, int* const pred) {
    if (blockIdx.x == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    int v1 = BLOCK_SIZE * blockId + idy;
    int v2 = BLOCK_SIZE * blockId + idx;

    __shared__ float cacheGraphBase[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePredBase[BLOCK_SIZE][BLOCK_SIZE];

    // Load base block for graph and predecessors
    int cellId = v1 * pitch + v2;

    if (v1 < nvertex && v2 < nvertex) {
        cacheGraphBase[idy][idx] = graph[cellId];
        cachePredBase[idy][idx] = pred[cellId];
    }
    else {
        cacheGraphBase[idy][idx] = MAX_DISTANCE;
        cachePredBase[idy][idx] = -1;
    }

    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        v2 = BLOCK_SIZE * blockIdx.x + idx;
    }
    else {
        // Load j-aligned singly dependent blocks
        v1 = BLOCK_SIZE * blockIdx.x + idy;
    }

    __shared__ float cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePred[BLOCK_SIZE][BLOCK_SIZE];

    // Load current block for graph and predecessors
    float currentPath;
    int currentPred;

    cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        currentPath = graph[cellId];
        currentPred = pred[cellId];
    }
    else {
        cacheGraphBase[idy][idx] = MAX_DISTANCE;
        cachePredBase[idy][idx] = -1;
    }
    cacheGraph[idy][idx] = currentPath;
    cachePred[idy][idx] = currentPred;

    // Synchronize to make sure the all value are saved in cache
    __syncthreads();

    float newPath;
    // Compute i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraphBase[idy][u] + cacheGraph[u][idx];

            if (newPath < currentPath) {
                currentPath = newPath;
                currentPred = cachePred[u][idx];
            }
            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

            // Update new values
            cacheGraph[idy][idx] = currentPath;
            cachePred[idy][idx] = currentPred;

            // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    }
    else {
        // Compute j-aligned singly dependent blocks
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];

            if (newPath < currentPath) {
                currentPath = newPath;
                currentPred = cachePredBase[u][idx];
            }

            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

            // Update new values
            cacheGraph[idy][idx] = currentPath;
            cachePred[idy][idx] = currentPred;

            // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = currentPath;
        pred[cellId] = currentPred;
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Independent phase 3
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_independent_ph(const int blockId, size_t pitch, const size_t nvertex, float* const graph, int* const pred) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;

    __shared__ float cacheGraphBaseRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float cacheGraphBaseCol[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePredBaseRow[BLOCK_SIZE][BLOCK_SIZE];

    int v1Row = BLOCK_SIZE * blockId + idy;
    int v2Col = BLOCK_SIZE * blockId + idx;

    // Load data for block
    int cellId;
    if (v1Row < nvertex && v2 < nvertex) {
        cellId = v1Row * pitch + v2;

        cacheGraphBaseRow[idy][idx] = graph[cellId];
        cachePredBaseRow[idy][idx] = pred[cellId];
    }
    else {
        cacheGraphBaseRow[idy][idx] = MAX_DISTANCE;
        cachePredBaseRow[idy][idx] = -1;
    }

    if (v1 < nvertex && v2Col < nvertex) {
        cellId = v1 * pitch + v2Col;
        cacheGraphBaseCol[idy][idx] = graph[cellId];
    }
    else {
        cacheGraphBaseCol[idy][idx] = MAX_DISTANCE;
    }

    // Synchronize to make sure the all value are loaded in virtual block
    __syncthreads();

    float currentPath;
    int currentPred;
    float newPath;

    // Compute data for block
    if (v1 < nvertex && v2 < nvertex) {
        cellId = v1 * pitch + v2;
        currentPath = graph[cellId];
        currentPred = pred[cellId];

#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
            if (currentPath > newPath) {
                currentPath = newPath;
                currentPred = cachePredBaseRow[u][idx];
            }
        }
        graph[cellId] = currentPath;
        pred[cellId] = currentPred;
    }
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param weight_matrix: Array of graph with distance between vertex on device, weight_ii = 0, MAX_DISTANCE if no path
 * @param pred_matrix: Array of predecessors for a graph on device, -1 if no path
 * @param distance_matrix: Array of shortest distance between vertex
 * @param n_vertex: Number of vertex
 */
void cudaBlockedFW(void* weight_matrix, void* pred_matrix, void* distance_matrix, size_t n_vertex, size_t* pitch) {
    HANDLE_ERROR(cudaSetDevice(0));
    int* pred_matrix_re = nullptr;
    // Copy weight matrix to result distance matrix
    HANDLE_ERROR(cudaMemcpy(distance_matrix, weight_matrix, n_vertex * n_vertex * sizeof(float), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMalloc(&pred_matrix_re, n_vertex * n_vertex * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(pred_matrix_re, distance_matrix, n_vertex * n_vertex * sizeof(int), cudaMemcpyDeviceToDevice));

    dim3 gridPhase1(1, 1, 1);
    dim3 gridPhase2((n_vertex - 1) / BLOCK_SIZE + 1, 2, 1);
    dim3 gridPhase3((n_vertex - 1) / BLOCK_SIZE + 1, (n_vertex - 1) / BLOCK_SIZE + 1, 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

    size_t numBlock = (n_vertex - 1) / BLOCK_SIZE + 1;

    // clock_t t1 = 0, t2 = 0, t3 = 0;
    for (int blockID = 0; blockID < numBlock; ++blockID) {
        // Start dependent phase
        // clock_t tmp1 = clock();
        _blocked_fw_dependent_ph << <gridPhase1, dimBlockSize >> >
            (blockID, n_vertex, n_vertex, (float*)distance_matrix, (int*)pred_matrix_re);
        // t1 += clock() - tmp1;

        // tmp1 = clock();
        // Start partially dependent phase
        _blocked_fw_partial_dependent_ph << <gridPhase2, dimBlockSize >> >
            (blockID, n_vertex, n_vertex, (float*)distance_matrix, (int*)pred_matrix_re);
        // t2 += clock() - tmp1;

        // tmp1 = clock();
        // Start independent phase
        _blocked_fw_independent_ph << <gridPhase3, dimBlockSize >> >
            (blockID, n_vertex, n_vertex, (float*)distance_matrix, (int*)pred_matrix_re);
        // t3 += clock() - tmp1;
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    // std::cout << t1 << "  " << t2 << "  " << t3 << "  " << std::endl;
}