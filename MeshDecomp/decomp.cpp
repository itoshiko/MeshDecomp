#include "Mesh.h"
#include <iostream>
#include "cu_common_util.cuh"
#include <thread>

int main()
{
	Mesh* mesh = nullptr;
	mesh = new Mesh("D:/course_proj/MeshDecomp/asset/dino.ply");
	mesh->preProcess();
	mesh->genFuzzyDecomp();
	mesh->genFinalDecomp(true);
	mesh->dumpFile("D:/course_proj/MeshDecomp/asset/decomp/dino.ply");
	return 0;
}


