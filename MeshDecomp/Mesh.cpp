#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Mesh.h"
#include "cu_common_util.cuh"
#include "floyd_warshall.cuh"
#include "flow.hpp"

Mesh::Mesh()
{
    num_verts = 0;
    num_faces = 0;
}

Mesh::Mesh(std::string file)
{
    num_verts = 0;
    num_faces = 0;
    miniply::PLYReader reader(file.c_str());
    if (!reader.valid()) {
        return;
    }

    uint32_t faceIdxs[3];
    miniply::PLYElement* faceElem = reader.get_element(reader.find_element(miniply::kPLYFaceElement));
    if (faceElem == nullptr) {
        return;
    }
    faceElem->convert_list_to_fixed_size(faceElem->find_property("vertex_indices"), 3, faceIdxs);

    uint32_t indexes[3];
    bool gotVerts = false, gotFaces = false;
    while (reader.has_element() && (!gotVerts || !gotFaces)) {
        if (reader.element_is(miniply::kPLYVertexElement) && reader.load_element() && reader.find_pos(indexes)) {
            num_verts = reader.num_rows();
            verts = new float[num_verts * 3];
            reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float, verts);
            gotVerts = true;
        }
        else if (!gotFaces && reader.element_is(miniply::kPLYFaceElement) && reader.load_element()) {
            num_faces = reader.num_rows();
            faces = new int[num_faces * 3];
            reader.extract_properties(faceIdxs, 3, miniply::PLYPropertyType::Int, faces);
            gotFaces = true;
        }
        if (gotVerts && gotFaces) {
            break;
        }
        reader.next_element();
    }
}

void Mesh::preProcess()
{
    // transfer vertices and faces data to GPU
    mapToDev();
    // calculate normal and center point of each faces
    if (face_normal == nullptr)
        HANDLE_ERROR(cudaMalloc(&face_normal, num_faces * 3 * sizeof(float)));
    computeFaceNormal(faces_dev, verts_dev, face_normal, num_faces);
    if (face_center == nullptr)
        HANDLE_ERROR(cudaMalloc(&face_center, num_faces * 3 * sizeof(float)));
    computeFaceCenter(faces_dev, verts_dev, face_center, num_faces);
    // test faces adjacency and form dual graph
    if (adj_pred_matrix == nullptr)
        HANDLE_ERROR(cudaMalloc(&adj_pred_matrix, num_faces * num_faces * sizeof(int)));
    if (graph_weight_matrix == nullptr)
        HANDLE_ERROR(cudaMalloc(&graph_weight_matrix, num_faces * num_faces * sizeof(float)));
    if (adj_cap_matrix == nullptr)
    {
        HANDLE_ERROR(cudaMalloc(&adj_cap_matrix, num_faces * num_faces * sizeof(float)));
        HANDLE_ERROR(cudaMemset(adj_cap_matrix, 0, num_faces * num_faces * sizeof(float)));
    }
    construct_adjacency(verts_dev, faces_dev, face_normal, face_center, adj_pred_matrix, graph_weight_matrix, adj_cap_matrix, delta, num_faces);

    //float* weight = new float[num_faces * num_faces];
    //cudaMemcpy(weight, graph_weight_matrix, num_faces * num_faces * sizeof(float), cudaMemcpyDeviceToHost);
    //float* _center = new float[3 * num_faces];
    //HANDLE_ERROR(cudaMemcpy(_center, face_center, num_faces * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    //std::ofstream f("D:/course_proj/MeshDecomp/debug/dual.obj");
    //for (int i = 0; i < num_faces; i++)
    //{
    //    f << "v" << " " << std::to_string(_center[i * 3]) << " " << std::to_string(_center[i * 3 + 1]) << " " << std::to_string(_center[i * 3 + 2]);
    //    f << " " << 1.0 << " " << 0.0 << " " << 0.0 << std::endl;
    //}
    //for (int i = 0; i < num_faces; i++)
    //{
    //    for (int j = 0; j < num_faces; j++)
    //    {
    //        if (weight[num_faces * i + j] < 99999.)
    //            f << "l " << i + 1 << " " << j + 1 << std::endl;
    //    }
    //}
    //f.close();
    //exit(0);


    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    HANDLE_ERROR(cudaMalloc(&adj_dist_matrix, num_faces * num_faces * sizeof(float)));
    // run ASAP using Floyd-Warshall
    cudaBlockedFW(graph_weight_matrix, adj_pred_matrix, adj_dist_matrix, num_faces, &dist_matrix_pitch);
}

Mesh::~Mesh()
{
    if (verts != nullptr) delete[] verts;
    if (faces != nullptr) delete[] faces;
}

void Mesh::mapToDev()
{
    if (verts == nullptr || faces == nullptr)
        return;
    if (verts_dev == nullptr)
        HANDLE_ERROR(cudaMalloc(&verts_dev, num_verts * 3 * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(verts_dev, verts, num_verts * 3 * sizeof(float), cudaMemcpyHostToDevice));
    if (faces_dev == nullptr)
        HANDLE_ERROR(cudaMalloc(&faces_dev, num_faces * 3 * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(faces_dev, faces, num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice));
}

void Mesh::genFuzzyDecomp(bool two)
{
    if (adj_dist_matrix == nullptr)
        return;
    int* reps = nullptr;
    if (two)
    {
        float* max_dist = nullptr;
        int* max_idx = nullptr;
        HANDLE_ERROR(cudaMalloc(&max_dist, sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&max_idx, sizeof(int)));
        ArrayArgmax(adj_dist_matrix, max_dist, max_idx, num_faces * num_faces, 999999.0);
        printf("Max distance: ");
        printDevice << <1, 1 >> > (max_dist, 1);
        reps = new int[2];
        int* max_idx_host = new int;
        HANDLE_ERROR(cudaMemcpy(max_idx_host, max_idx, sizeof(int), cudaMemcpyDeviceToHost));
        reps[0] = *max_idx_host / num_faces;
        reps[1] = *max_idx_host % num_faces;
        k_rep = 2;
    }
    else
    {
        reps = new int[max_rep_k];
        k_rep = search_reps_k(adj_dist_matrix, reps, num_faces, max_rep_k);
    }

    std::cout << "Representatives retrieved, number: " << k_rep << std::endl;
    for (int i = 0; i < k_rep; i++)
    {
        std::cout << " " << i + 1 << ": id=" << reps[i] << ", coord(";
        std::cout << verts[faces[reps[i] * 3] * 3] << "  " << verts[faces[reps[i] * 3] * 3 + 1] << "  " << verts[faces[reps[i] * 3] * 3 + 2];
        std::cout << ")" << std::endl;
    }

    // update representation
    float* prob_matrix = nullptr;
    float* matric_matrix = nullptr;
    HANDLE_ERROR(cudaMalloc(&prob_matrix, k_rep * num_faces * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&matric_matrix, k_rep * num_faces * sizeof(float)));

    // get face classification
    if (faces_type == nullptr)
        HANDLE_ERROR(cudaMalloc(&faces_type, 2 * num_faces * sizeof(int)));

    for (int iter = 0; iter < REP_ITER_MAX; iter++)
    {
        bool updated = update_representation(adj_dist_matrix, prob_matrix, matric_matrix, faces_type, reps, 0.05, k_rep, num_faces);
        //cudaMemcpy(prob_debug, prob_matrix, num_faces * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        //std::string path("D:/course_proj/MeshDecomp/debug/face_");
        //path += std::to_string(iter + 1);
        //path += ".obj";
        //debugFcaceProperty(prob_debug, path, true);
        if (!updated)
            break;
        std::cout << "UPDATE ITER " << iter << std::endl;
        for (int i = 0; i < k_rep; i++)
        {
            std::cout << " " << i + 1 << ": id=" << reps[i] << ", coord(";
            std::cout << verts[faces[reps[i] * 3] * 3] << "  " << verts[faces[reps[i] * 3] * 3 + 1] << "  " << verts[faces[reps[i] * 3] * 3 + 2];
            std::cout << ")" << std::endl;
        }
    }
}

void Mesh::genFinalDecomp()
{
    // prepare for max-flow
    int _n = num_faces;
    Dinic** graphs = new Dinic *[k_rep - 1];
    for (int g1 = 0; g1 < k_rep - 1; g1++) {
        graphs[g1] = (Dinic*)operator new((k_rep - g1 - 1) * sizeof(Dinic));
        for (int g2 = 0; g2 < k_rep - g1 - 1; g2++)
            new (graphs[g1] + g2) Dinic(num_faces + 2);
    }
    // download flow capacity and patch information to CPU
    float* graph_cap_host = new float[num_faces * num_faces];
    type_host = new int[num_faces * k_rep * sizeof(int)];
    HANDLE_ERROR(cudaMemcpy(graph_cap_host, adj_cap_matrix, num_faces * num_faces * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(type_host, faces_type, num_faces * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    int* cnt = new int[k_rep];
    memset(cnt, 0, k_rep * sizeof(int));
    for (int i = 0; i < num_faces; i++)
    {
        if (type_host[i + num_faces] < 0)
            cnt[type_host[i]]++;
    }
    for (int i = 0; i < k_rep; i++)
        printf("rep %d  faces cnt %d\n", i, cnt[i]);

    // Construct flow network graphs
    for (int i = 0; i < num_faces; i++)
    {
        for (int j = i + 1; j < num_faces; j++)
        {
            // patch that i belongs to has been determined
            if (type_host[num_faces + i] < 0)
            {
                if (graph_cap_host[i * num_faces + j] > TOL)
                {
                    // if the other node (j) that this edge connects fuzzy, add both (i, j), (j, i), (i, sink)
                    if (type_host[num_faces + j] >= 0)
                    {
                        int gid_1 = min(type_host[num_faces + j], type_host[j]);
                        int gid_2 = max(type_host[num_faces + j], type_host[j]) - gid_1 - 1;
                        graphs[gid_1][gid_2].AddEdge(i + 1, j + 1, graph_cap_host[i * num_faces + j]);  // (i -> j)
                        graphs[gid_1][gid_2].AddEdge(j + 1, i + 1, graph_cap_host[i * num_faces + j]);  // (j -> i)
                        if (type_host[i] == gid_1)
                        {
                            graphs[gid_1][gid_2].AddEdge(num_faces + 2, i + 1, 99999.);  // (source -> i)
                        }
                        else
                        {
                            graphs[gid_1][gid_2].AddEdge(i + 1, num_faces + 1, 99999.);  // (i -> sink)
                        }
                    }
                }
            }
            // patch that i belongs is not determined
            else
            {
                if (graph_cap_host[i * num_faces + j] > TOL)
                {
                    // if the other node (j) that this edge connects also fuzzy, add both (i, j), (j, i)
                    if (type_host[num_faces + j] >= 0)
                    {
                        int gid_1 = min(type_host[num_faces + i], type_host[i]);
                        int gid_2 = max(type_host[num_faces + i], type_host[i]) - gid_1 - 1;
                        graphs[gid_1][gid_2].AddEdge(i + 1, j + 1, graph_cap_host[i * num_faces + j]);  // (i -> j)
                        graphs[gid_1][gid_2].AddEdge(j + 1, i + 1, graph_cap_host[i * num_faces + j]);  // (j -> i)
                    }
                    // if the other node (j) that this edge connects not fuzzy, add both (i, j), (j, i), (source, j)
                    else
                    {
                        int gid_1 = min(type_host[num_faces + i], type_host[i]);
                        int gid_2 = max(type_host[num_faces + i], type_host[i]) - gid_1 - 1;
                        graphs[gid_1][gid_2].AddEdge(i + 1, j + 1, graph_cap_host[i * num_faces + j]);  // (i -> j)
                        graphs[gid_1][gid_2].AddEdge(j + 1, i + 1, graph_cap_host[i * num_faces + j]);  // (j -> i)
                        if (type_host[j] == gid_1)
                        {
                            graphs[gid_1][gid_2].AddEdge(num_faces + 2, j + 1, 99999.);  // (source -> j)
                        }
                        else
                        {
                            graphs[gid_1][gid_2].AddEdge(j + 1, num_faces + 1, 99999.);  // (j -> sink)
                        }
                    }
                }
            }
        }

    }
    
    // Solove max-flow min-cut problem and get final decomposition
    for (int g1 = 0; g1 < k_rep - 1; g1++) {
        for (int g2 = 0; g2 < k_rep - g1 - 1; g2++) {
            std::vector<int> class_g1;
            graphs[g1][g2].cut_classify(num_faces + 2, num_faces + 1, class_g1);
            for (auto _v : class_g1)
            {
                type_host[_v - 1] = g1;
                type_host[_v - 1 + num_faces] = -1;
            }
        }
    }
    for (int i = 0; i < num_faces; i++)
    {
        if (type_host[i + num_faces] < 0)
        {
            type_host[i] = max(type_host[num_faces + i], type_host[i]);
            type_host[i + num_faces] = -1;
        }
    }
}

void Mesh::dumpFile(std::string path)
{
    std::ofstream f(path);
    // Write to the file
    float* _center = new float[3 * num_faces];
    HANDLE_ERROR(cudaMemcpy(_center, face_center, num_faces * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_faces; i++)
    {
        f << "v" << " " << std::to_string(_center[i * 3]) << " " << std::to_string(_center[i * 3 + 1]) << " " << std::to_string(_center[i * 3 + 2]);
        switch (type_host[i])
        {
        case 0:
            f << " 0.918 0.2 0.14" << std::endl;
            break;
        case 1:
            f << " 0.6289 0.99 0.34" << std::endl;
            break;
        case 2:
            f << " 0.2 0.52 0.99" << std::endl;
            break;
        case 3:
            f << " 0.96 0.52 0.3" << std::endl;
            break;
        case 4:
            f << " 0.2 0.5 0.5" << std::endl;
            break;
        case 5:
            f << " 0.46 0.095 0.25" << std::endl;
            break;
        case 6:
            f << " 1.0 0.99 0.33" << std::endl;
            break;
        case 7:
            f << " 0.1 0.25 0.05" << std::endl;
            break;
        case 8:
            f << " 0.914 0.195 0.5" << std::endl;
            break;
        case 9:
            f << " 0.418 0.297 0.23" << std::endl;
            break;
        case 10:
            f << " 0.49 0.51 0.124" << std::endl;
            break;
        case 11:
            f << " 0.45 0.168 0.957" << std::endl;
            break;
        }
    }
    // Close the file
    f.close();
}

void Mesh::debugFcaceProperty(float* prop, std::string path, bool normalize)
{
    std::ofstream f(path);
    // Write to the file
    float* _center = new float[3 * num_faces];
    HANDLE_ERROR(cudaMemcpy(_center, face_center, num_faces * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_faces; i++)
    {
        f << "v" << " " << std::to_string(_center[i * 3]) << " " << std::to_string(_center[i * 3 + 1]) << " " << std::to_string(_center[i * 3 + 2]);
        if (prop[i * 2] > 0.55)
            f << " " << 1.0 << " " << 0.0 << " " << 0.0 << std::endl;
        else if (prop[i * 2] < 0.45)
            f << " " << 0.0 << " " << 1.0 << " " << 0.0 << std::endl;
        else
            f << " " << 0.0 << " " << 1.0 << " " << 1.0 << std::endl;
    }
    // Close the file
    f.close();
}
