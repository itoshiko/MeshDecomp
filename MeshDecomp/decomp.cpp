#include "Mesh.h"
#include <iostream>
#include "cu_common_util.cuh"
#include <thread>

int main()
{
	Mesh mesh("D:/course_proj/MeshDecomp/asset/smpl_mesh.ply");
	mesh.preProcess();
	mesh.genFuzzyDecomp(true);
	mesh.genFinalDecomp(2);
	mesh.dumpFile("D:/course_proj/MeshDecomp/debug/decomp.obj");
	std::cout << "test\n";
}


