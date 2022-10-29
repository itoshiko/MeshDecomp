#ifndef _MESH_UTIL_
#define _MESH_UTIL_

#include "helper_math.h"
#include <vector_functions.h>

#define TOL 1e-6

void computeFaceNormal(
    int3* faces, 
    float3* verts, 
    float3* normals, 
    size_t cnt);

void computeFaceCenter(
    int3* faces, 
    float3* verts, 
    float3* centers, 
    size_t cnt);

void construct_adjacency(
    float3* verts,
    int3* faces,
    float3* normals,
    float3* centers,
    int* pred,
    float* graph,
    float* flow,
    float delta,
    size_t face_num);

int search_reps_k(
    float* dist_matrix, 
    int* reps, 
    size_t face_num, 
    int max_rep);

bool update_representation(
    float* dist_matrix,
    float* prob_matrix,
    float* matric_matrix,
    int* type,
    int* reps,
    float eps,
    int k_rep,
    size_t face_num);

void get_face_label(
    float* prob_matrix,
    int* type1,
    int* type2,
    bool* mask,
    int k_rep,
    float eps,
    size_t face_num);

#endif
