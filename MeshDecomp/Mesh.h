#pragma once
#include <string>
#include <fstream>
#include "miniply.h"
#include "mesh_util.cuh"

#define REP_ITER_MAX 20

class Mesh
{
public:
	Mesh();
	Mesh(std::string file);
	void preProcess();
	void genFuzzyDecomp(bool two);
	void genFinalDecomp();
	void dumpFile(std::string path);
	void debugFcaceProperty(float* prop, std::string path, bool normalize);
	~Mesh();
private:
	// functions
	void mapToDev();
	
	// parameters
	size_t dist_matrix_pitch = 0;
	float delta = 0.8;
	int max_rep_k = 20;
	int k_rep = -1;

	// data field
	size_t num_verts;
	size_t num_faces;
	float* verts = nullptr;
	int* faces = nullptr;
	float3* verts_dev = nullptr;
	int3* faces_dev = nullptr;
	float3* face_normal = nullptr;
	float3* face_center = nullptr;
	int* adj_pred_matrix = nullptr;
	float* graph_weight_matrix = nullptr;
	float* adj_dist_matrix = nullptr;
	float* adj_cap_matrix = nullptr;
	int* faces_type = nullptr;
	int* type_host = nullptr;
};

