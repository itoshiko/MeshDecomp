#include "Mesh.h"
#include <iostream>
#include "cu_common_util.cuh"
#include <thread>


int main(int argc, char** argv)
{
	if (argc < 3)
		printf("Parameter format: input, output, recursive (optional, 0 for single decomposition, 1 for recursive)\n");
	bool recursive = true;
	if (argc > 3 && argv[3][0] == '0') recursive = false;
	Mesh* mesh = nullptr;
	mesh = new Mesh(argv[1]);
	mesh->preProcess();
	mesh->genFuzzyDecomp();
	mesh->genFinalDecomp(recursive);
	mesh->dumpFile(argv[2]);
	return 0;
}


