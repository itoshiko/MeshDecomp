#pragma once
#include <string>
#include <fstream>
#include "miniply.h"
#include "mesh_util.cuh"

#define REP_ITER_MAX 20
#define TH_DIST_RATIO 0.25
#define TH_ANGLE_DIFF 0.30
#define TH_REP_DIST 0.15

class Mesh
{
public:
	Mesh();
	Mesh(std::string file);
	void preProcess();
	void genFuzzyDecomp();
	void genFinalDecomp(bool recur);
	void dumpFile(std::string path);
	void debugFcaceProperty(float* prop, std::string path, bool normalize);
	~Mesh();
private:
	// functions
	void mapToDev();
	void decompRecur();
	int* getClassification() { return type_host; };
	int getRepNum() { return k_rep; }
	Mesh(Mesh* mesh, std::vector<int>& select);
	void debugNormal();
	
	// parameters
	size_t dist_matrix_pitch = 0;
	float delta = 0.8;
	int max_rep_k = 20;
	int k_rep = -1;
	bool two = false;
	bool master = true;
	int max_cnt = 0;

	float max_patch_dist = -1.;
	float* patch_avg_dist = nullptr;
	float global_avg_dist = 0.;
	float dihedral_ang_diff = 0.;
	float face_max_dist = 0.;

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

