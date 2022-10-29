#include "Mesh.h"
#include <iostream>
#include "cu_common_util.cuh"
#include <thread>

int main()
{
	Mesh mesh("D:/course_proj/MeshDecomp/asset/smpl_mesh_5k.ply");
	mesh.preProcess();
	mesh.genFuzzyDecomp(false);
	mesh.genFinalDecomp();
	mesh.dumpFile("d:/course_proj/meshdecomp/debug/decomp_5k.obj");
	std::cout << "test\n";
}


