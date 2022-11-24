#include "Mesh.h"
#include <iostream>
#include "cu_common_util.cuh"
#include <thread>

int main()
{
	Mesh mesh("D:/course_proj/MeshDecomp/asset/smpl_mesh.ply");
	// Mesh mesh("D:/course_proj/Mesh-Segmentation-master/dino_fuzzyK1_010_new.ply");
	mesh.preProcess();
	mesh.genFuzzyDecomp();
	mesh.genFinalDecomp(true);
	mesh.dumpFile("d:/course_proj/meshdecomp/debug/decomp_k_r.ply");
	return 0;
}


