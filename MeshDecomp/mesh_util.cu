#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mesh_util.cuh"
#include "cu_common_util.cuh"
#include <cub/cub.cuh>

static __global__
void _compute_face_normal(int3* faces, float3* verts, float3* normals, size_t cnt) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int fid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (!(fid < cnt)) return;
    // calculate face normal
    int3 face = faces[fid];
    float3 v0 = verts[face.x];
    float3 v1 = verts[face.y];
    float3 v2 = verts[face.z];
    normals[fid] = normalize(cross(v1 - v0, v2 - v0));
}

void computeFaceNormal(int3* faces, float3* verts, float3* normals, size_t cnt)
{
    int BLOCK_SIZE = 16;
    int _grid = int(sqrt(double((cnt - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1))) + 1;
    dim3 gridSize(_grid, _grid, 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    _compute_face_normal << <gridSize, dimBlockSize >> >
        (faces, verts, normals, cnt);
}

static __global__
void _compute_face_center(int3* faces, float3* verts, float3* centers, size_t cnt) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int fid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (!(fid < cnt)) return;
    // calculate face ccenter point
    int3 face = faces[fid];
    centers[fid] = (verts[face.x] + verts[face.y] + verts[face.z]) / 3.0;
}

void computeFaceCenter(int3* faces, float3* verts, float3* centers, size_t cnt)
{
    int BLOCK_SIZE = 16;
    int _grid = int(sqrt(double((cnt - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1))) + 1;
    dim3 gridSize(_grid, _grid, 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    _compute_face_center << <gridSize, dimBlockSize >> >
        (faces, verts, centers, cnt);
}

static __device__ __forceinline__
float _compute_face_geo_dis(float3 p0, float3 p1, float3 ca, float3 cb)
{
    float3 p0_p1 = p1 - p0;
    float3 p0_ca = ca - p0;
    float3 p0_cb = cb - p0;
    float l_p0_p1 = length(p0_p1);
    float l_p0_ca = length(p0_ca);
    float l_p0_cb = length(p0_cb);
    float ang1 = acosf(dot(p0_ca, p0_p1) / l_p0_p1 / l_p0_ca);
    float ang2 = acosf(dot(p0_cb, p0_p1) / l_p0_p1 / l_p0_cb);
    float geo_d = l_p0_ca * l_p0_ca + l_p0_cb * l_p0_cb - 2 * l_p0_ca * l_p0_cb * cosf(ang1 + ang2);
    return geo_d;
}

/**
* given faces A and B, the one vertex in B that is not shared with A, projected onto the plane of A
* has a projection that is zero or negative.
*/
static __device__ __forceinline__
float _convex_indicator(float3 normal_a, float3 p_shared, float3 p_unshared_b)
{
    // normals from the first column of face adjacency
    float3 vector_other = p_unshared_b - p_shared;
    float proj = dot(vector_other, normal_a);
    // get the projection with a dot product
    return ((proj < TOL) ? 0.2 : 1.0);
}

static __device__ __forceinline__
float _compute_face_ang_dis(float3 normal_a, float3 normal_b, float3 p_shared, float3 p_unshared_b)
{
    float ind = _convex_indicator(normal_a, p_shared, p_unshared_b);
    float ang_d = 1. - dot(normal_a, normal_b);
    return ind * ang_d;
}

static __global__
void _construct_face_adjacency_phase1(
    float3* verts,
    int3* faces,
    float3* normals,
    float3* centers,
    float* ang_dis_matrix,
    float* geo_dis_matrix,
    int* pred,
    int* adj,
    size_t face_num,
    size_t pitch)
{
    int f1_id = blockIdx.x * blockDim.x + threadIdx.x;
    int f2_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (!((f1_id < face_num) && (f2_id < face_num)) || f1_id >= f2_id)
        return;
    if (f1_id == f2_id)
    {
        pred[pitch * f1_id + f2_id] = -1;
        ang_dis_matrix[pitch * f1_id + f2_id] = 0;
        geo_dis_matrix[pitch * f1_id + f2_id] = 0;
        adj[pitch * f1_id + f2_id] = 0;
        return;
    }
    // compute if f1 and f2 is adjacent, if yes, calculate their distance
    int3 f1 = faces[f1_id];
    int3 f2 = faces[f2_id];
    int share_1 = -1, share_2 = -1;
    int3 tmp1 = f1 - f2.x;
    if (tmp1.x == 0 || tmp1.y == 0 || tmp1.z == 0)
        share_1 = f2.x;
    int3 tmp2 = f1 - f2.y;
    if (tmp2.x == 0 || tmp2.y == 0 || tmp2.z == 0)
    {
        if (share_1 < 0) share_1 = f2.y;
        else share_2 = f2.y;
    }
    int3 tmp3 = f1 - f2.z;
    if (tmp3.x == 0 || tmp3.y == 0 || tmp3.z == 0)
    {
        if (share_1 < 0) share_1 = f2.z;
        else share_2 = f2.z;
    }
    // no shared edge
    if (share_1 < 0 || share_2 < 0)
    {
        pred[pitch * f1_id + f2_id] = -1;
        pred[pitch * f2_id + f1_id] = -1;
        ang_dis_matrix[pitch * f1_id + f2_id] = 0;
        ang_dis_matrix[pitch * f2_id + f1_id] = 0;
        geo_dis_matrix[pitch * f1_id + f2_id] = 0;
        geo_dis_matrix[pitch * f2_id + f1_id] = 0;
        adj[pitch * f1_id + f2_id] = 0;
        adj[pitch * f2_id + f1_id] = 0;
        return;
    }
    int unshare_b = -1;
    if (f2.x != share_1 && f2.x != share_2) unshare_b = f2.x;
    else if (f2.y != share_1 && f2.y != share_2) unshare_b = f2.y;
    else unshare_b = f2.z;

    // assign distance matrix
    float geo_dis = _compute_face_geo_dis(verts[share_1], verts[share_2], centers[f1_id], centers[f2_id]);
    geo_dis_matrix[pitch * f1_id + f2_id] = geo_dis;
    geo_dis_matrix[pitch * f2_id + f1_id] = geo_dis;
    float ang_dis = _compute_face_ang_dis(normals[f1_id], normals[f2_id], verts[share_1], verts[unshare_b]);
    ang_dis_matrix[pitch * f1_id + f2_id] = ang_dis;
    ang_dis_matrix[pitch * f2_id + f1_id] = ang_dis;
    pred[pitch * f1_id + f2_id] = f1_id;
    pred[pitch * f2_id + f1_id] = f2_id;
    adj[pitch * f1_id + f2_id] = 1;
    adj[pitch * f2_id + f1_id] = 1;
}

static __global__
void _construct_face_adjacency_phase2(
    float* ang_dis_matrix,
    float* geo_dis_matrix,
    float* ang_dis_avg,
    float* geo_dis_avg,
    float* graph,
    float* flow,
    int* adj,
    float delta,
    size_t face_num,
    size_t pitch)
{
    int f1_id = blockIdx.x * blockDim.x + threadIdx.x;
    int f2_id = blockIdx.y * blockDim.y + threadIdx.y;
    int cid_1 = pitch * f1_id + f2_id;
    int cid_2 = pitch * f2_id + f1_id;
    if (!((f1_id < face_num) && (f2_id < face_num)) || f1_id > f2_id)
        return;
    if (f1_id == f2_id)
    {
        graph[cid_1] = 0.;
        return;
    }
    if (adj[cid_1] < TOL)
    {
        graph[cid_1] = 10000000.0;
        graph[cid_2] = 10000000.0;
        return;
    }
    // geo_dis_matrix[cid_1] /= geo_dis_avg[0];
    float weight = delta * (geo_dis_matrix[cid_1] / geo_dis_avg[0]);
    // ang_dis_matrix[cid_1] /= ang_dis_avg[0];
    weight += (1. - delta) * (ang_dis_matrix[cid_1] / ang_dis_avg[0]);
    // printf("%f ", ang_dis_matrix[cid_1]);
    flow[cid_1] = ang_dis_avg[0] / (ang_dis_avg[0] + ang_dis_matrix[cid_1]);
    graph[cid_1] = weight;
    graph[cid_2] = weight;
}

void construct_adjacency(
    float3* verts,
    int3* faces,
    float3* normals,
    float3* centers, 
    int* pred,
    float* graph,
    float* flow,
    float delta,
    size_t face_num)
{
    int BLOCK_SIZE_1 = 16, BLOCK_SIZE_2 = 32;
    dim3 gridSize((face_num - 1) / BLOCK_SIZE_1 + 1, (face_num - 1) / BLOCK_SIZE_2 + 1, 1);
    dim3 blockSize(BLOCK_SIZE_1, BLOCK_SIZE_2, 1);

    float* ang_dis_matrix = nullptr;
    float* geo_dis_matrix = nullptr;
    int* adj = nullptr;
    HANDLE_ERROR(cudaMalloc(&ang_dis_matrix, face_num * face_num * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&geo_dis_matrix, face_num * face_num * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&adj, face_num * face_num * sizeof(int)));
    _construct_face_adjacency_phase1 << <gridSize, blockSize >> >
        (verts, faces, normals, centers, ang_dis_matrix, geo_dis_matrix, pred, adj, face_num, face_num);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    float* geo_dist_avg = nullptr;
    float* ang_dist_avg = nullptr;
    int* valid_cell_cnt = nullptr;
    HANDLE_ERROR(cudaMalloc(&geo_dist_avg, sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&ang_dist_avg, sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&valid_cell_cnt, sizeof(int)));
    ArrayReduceSum(geo_dis_matrix, geo_dist_avg, face_num * face_num);
    ArrayReduceSum(ang_dis_matrix, ang_dist_avg, face_num * face_num);
    ArrayReduceSum(adj, valid_cell_cnt, face_num * face_num);
    //printDevice << <1, 1 >> > (ang_dist_avg, 1);
    //printDevice << <1, 1 >> > (valid_cell_cnt, 1);
    //printDevice << <1, 1 >> > (geo_dist_avg, 1);
    selfDivide << <1, 1 >> > (geo_dist_avg, valid_cell_cnt);
    selfDivide << <1, 1 >> > (ang_dist_avg, valid_cell_cnt);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _construct_face_adjacency_phase2 << <gridSize, blockSize >> > 
        (ang_dis_matrix, geo_dis_matrix, ang_dist_avg, geo_dist_avg, graph, flow, adj, delta, face_num, face_num);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}

static __global__
void _calc_prob(float* dist_matrix, float* prob_matrix, int* reps, int k_rep, size_t face_num) {
    int face_id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (face_id >= face_num)
        return;
    float denominator = 0.;
#pragma unroll
    for (int r = 0; r < k_rep; r++)
    {
        denominator += 1. / (dist_matrix[face_id * face_num + reps[r]] + 1e-6);
    }
#pragma unroll
    for (int r = 0; r < k_rep; r++)
    {
        prob_matrix[face_id * k_rep + r] = (1. / (dist_matrix[face_id * face_num + reps[r]] + 1e-6)) / denominator;
    }
    // printf("face %d rep %d  %f\n", face_id, rep_id, prob_matrix[face_id * k_rep + rep_id]);
}

template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
static __global__
void _calc_patch_avg_dis(float* dist_matrix, float* avg_dist, int* type, size_t face_num) {
    int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;
    int face_id = blockIdx.x + blockIdx.y * gridDim.x;
    int rep_id = blockIdx.z;
    if (face_id < face_num)
    {
        float _sum = 0.;
        int _cnt = 0;
        // Block wise reduction so that one thread in each block holds sum of thread results
        typedef cub::BlockReduce<float, BLOCK_DIM_X, cub::BLOCK_REDUCE_RAKING, BLOCK_DIM_Y> BlockReduceF;
        typedef cub::BlockReduce<int, BLOCK_DIM_X, cub::BLOCK_REDUCE_RAKING, BLOCK_DIM_Y> BlockReduceI;
        __shared__ typename BlockReduceF::TempStorage temp_storage_sum;
        __shared__ typename BlockReduceI::TempStorage temp_storage_cnt;
#pragma unroll
        for (int idx = threadId; idx < face_num; idx += BLOCK_DIM_X * BLOCK_DIM_Y)
        {
            // if (idx) belongs to patch (rep_id)
            if (type[idx] == rep_id && type[idx + face_num] < 0)
            {
                _sum += dist_matrix[face_id * face_num + idx];
                _cnt += 1;
            }
        }
        float aggregate = BlockReduceF(temp_storage_sum).Sum(_sum);
        int total_cnt = BlockReduceI(temp_storage_cnt).Sum(_cnt);
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            avg_dist[face_num * rep_id + face_id] = aggregate / (float)total_cnt;
        }
    }
}

static __global__
void _recalc_prob(float* avg_dist_patch, float* prob_matrix, int k_rep, size_t face_num) {
    int face_id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (face_id >= face_num)
        return;
    float denominator = 0.;
#pragma unroll
    for (int r = 0; r < k_rep; r++)
    {
        denominator += 1. / (avg_dist_patch[face_num * r + face_id] + 1e-6);
    }
#pragma unroll
    for (int r = 0; r < k_rep; r++)
    {
        prob_matrix[face_id * k_rep + r] = (1. / (avg_dist_patch[face_num * r + face_id] + 1e-6)) / denominator;
    }
}

/**
 * Calculation of matrices of faces to determine representation. 
 * A single thread do: sum(matric'(face_i, face_j, rep_k)), j = threadId, threadId + threadPerBlk, ..., i, k fixed
 * A single block do: matric(face_i, rep_k) = sum(matric'(face_i, face_j)), j = 0, ..., face_num, i, k fixed
 * Grid[i, j, k] do: matice(face_{ij}, rep_z), {ij} = 0, ..., face_num, k fixed
 *
 * @param dist_matrix: Array of graph with SP between vertex on device, [face_num * face_num]
 * @param prob_matrix: Array of probability which patch faces belong to, [face_0(rep_0, ..., rep_k), ..., face_n(rep_0, ..., rep_k)]
 * @param matric: Result array, [rep_0(face_0, ..., face_n), ..., rep_k(face_0, ..., face_n)]
 * @param k_rep: Number of representation
 * @param face_num: Number of faces
 */
template<int BLOCK_SIM_X, int BLOCK_DIM_Y>
static __global__
void _calc_face_matric(float* dist_matrix, float* prob_matrix, float* matric, int k_rep, size_t face_num) {
    int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;
    int face_id = blockIdx.x + blockIdx.y * gridDim.x;
    int rep_id = blockIdx.z;
    if (face_id < face_num)
    {
#pragma unroll
        float _sum = 0.;
        // Block wise reduction so that one thread in each block holds sum of thread results
        typedef cub::BlockReduce<float, BLOCK_SIM_X, cub::BLOCK_REDUCE_RAKING, BLOCK_DIM_Y> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        for (int idx = threadId; idx < face_num; idx += BLOCK_SIM_X * BLOCK_DIM_Y)
        {
            _sum += dist_matrix[face_id * face_num + idx] * prob_matrix[idx * k_rep + rep_id];
        }
        float aggregate = BlockReduce(temp_storage).Sum(_sum);
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            matric[face_id + rep_id * face_num] = aggregate;
            // printf("idx %d  %f\n", face_id + k_rep * face_num, matric[face_id + rep_id * face_num]);
        }
    }
}

bool update_representation(float* dist_matrix, float* prob_matrix, float* matric_matrix, int* type, int* reps, float eps, int k_rep, size_t face_num)
{
    int _grid = int(sqrt(double(face_num))) + 1;
    dim3 dimGridPhase1((face_num - 1) / (32 * 16) + 1, 1);
    dim3 dimGridPhase2(_grid, _grid, k_rep);
    dim3 dimBlockSize(32, 16, 1);

    // first phase: probability calculation
    int* reps_dev = nullptr;
    HANDLE_ERROR(cudaMalloc(&reps_dev, sizeof(int) * k_rep));
    HANDLE_ERROR(cudaMemcpy(reps_dev, reps, sizeof(int) * k_rep, cudaMemcpyHostToDevice));
    _calc_prob << <dimGridPhase1, dimBlockSize >> > (dist_matrix, prob_matrix, reps_dev, k_rep, face_num);

    // second phase: assign label and recompute probability according to method stated in paper
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    get_face_label(prob_matrix, type, type + face_num, k_rep, eps, face_num);
    float* avg_dist = nullptr;
    HANDLE_ERROR(cudaMalloc(&avg_dist, face_num * k_rep * sizeof(float)));
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _calc_patch_avg_dis<32, 16> << <dimGridPhase2, dimBlockSize >> > (dist_matrix, avg_dist, type, face_num);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _recalc_prob << <dimGridPhase1, dimBlockSize >> > (avg_dist, prob_matrix, k_rep, face_num);

    // third phase: matrices calculation
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _calc_face_matric<32, 16> << <dimGridPhase2, dimBlockSize >> > (dist_matrix, prob_matrix, matric_matrix, k_rep, face_num);
    // _calc_face_matric<1> << <dimGridPhase2, 1 >> > (dist_matrix, prob_matrix, matric_matrix, k_rep, face_num);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    bool updated = false;
    float* min_val = nullptr;
    HANDLE_ERROR(cudaMalloc(&min_val, sizeof(float)));
    for (int rid = 0; rid < k_rep; rid++)
    {
        int rep_prev = reps[rid];
        // printDevice << <1, 1 >> > (matric_matrix + rid * face_num, face_num);
        ArrayArgmin(matric_matrix + rid * face_num, min_val, reps_dev + rid, face_num);
        printDevice << <1, 1 >> > (min_val, 1);
        // printDevice << <1, 1 >> > (reps_dev + rid, 1);
        HANDLE_ERROR(cudaMemcpy(reps + rid, reps_dev + rid, sizeof(int), cudaMemcpyDeviceToHost));
        if (rep_prev != reps[rid])
            updated = true;
    }
    if (!updated)
    {
        printf("Converge!\n");
        get_face_label(prob_matrix, type, type + face_num, k_rep, eps, face_num);
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        _calc_patch_avg_dis<32, 16> << <dimGridPhase2, dimBlockSize >> > (dist_matrix, avg_dist, type, face_num);
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        _recalc_prob << <dimGridPhase1, dimBlockSize >> > (avg_dist, prob_matrix, k_rep, face_num);
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        get_face_label(prob_matrix, type, type + face_num, k_rep, eps, face_num);
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
    return updated;
}

static __device__
void _serach_top_2(float* data, int num, float* result_val, int* result_idx)
{
    float tmp = -999999.;
    int tmp_idx = -1;
    for (int i = 0; i < num; i++)
    {
        if (data[i] > tmp)
        {
            tmp = data[i];
            tmp_idx = i;
        }
    }
    result_val[0] = tmp;
    result_idx[0] = tmp_idx;
    tmp = -999999.;
    tmp_idx = -1;
#pragma unroll
    for (int i = 0; i < num; i++)
    {
        if ((data[i] > tmp) && (i != result_idx[0]))
        {
            tmp = data[i];
            tmp_idx = i;
        }
    }
    result_val[1] = tmp;
    result_idx[1] = tmp_idx;
}

static __global__
void _classify_vertex(float* prob_matrix, int* type1, int* type2, int k_rep, float eps, size_t cnt)
{
    int vid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (vid >= cnt)
        return;
    int top2_idx[2];
    float top2_prob[2];
    _serach_top_2(prob_matrix + vid * k_rep, k_rep, top2_prob, top2_idx);
    if (top2_prob[0] > 0.5 + eps)
    {
        type1[vid] = top2_idx[0];
        type2[vid] = -1;
        return;
    }
    type1[vid] = top2_idx[0];
    type2[vid] = top2_idx[1];
}

void get_face_label(float* prob_matrix, int* type1, int* type2, int k_rep, float eps, size_t face_num)
{
    int BLOCK_SIZE_1 = 16, BLOCK_SIZE_2 = 32;
    dim3 blockSize(BLOCK_SIZE_1, BLOCK_SIZE_2, 1);
    _classify_vertex << <(face_num - 1) / (BLOCK_SIZE_1 * BLOCK_SIZE_2) + 1, blockSize >> >
        (prob_matrix, type1, type2, k_rep, eps, face_num);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}

