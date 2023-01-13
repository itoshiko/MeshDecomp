# Mesh Decomposition
Implementation of Hierarchical Mesh Decomposition using Fuzzy Clustering and Cuts(https://dl.acm.org/doi/10.1145/1201775.882369). [also a course project of "Computer Graphics (计算机图形学)" of CS, Tsinghua University]

Some processes implemented by naïve CUDA kernels for acceleration, but not carefully optimized. Achieving a total ~5x acceleration compared with baseline (https://github.com/fornorp/Mesh-Segmentation). Performing hierarchical k-way decomposition on a mesh containing 13000+ faces takes about 5s with RTX 3090.

Usage
MeshDecomp.exe <input> <output> <recursive (optional, 0 for single decomposition, 1 for recursive(default))>

Reference

* CUDA blocked Floyd Warshall implementation by https://github.com/MTB90/cuda-floyd_warshall
* Dinic max flow implementation by https://oi-wiki.org/graph/flow/max-flow/
* ply file reader by https://github.com/vilya/miniply

