#include "cu_common_util.cuh"
#include <cub/cub.cuh>


void ArrayReduceSum(const float* input, float* output, size_t n) {
    size_t temp_storage_bytes;
    float* temp_storage = NULL;
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, input, output, n);
    HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
    cudaDeviceSynchronize();
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, input, output, n);
    cudaDeviceSynchronize();
}

void ArrayReduceSum(const int* input, int* output, size_t n) {
    size_t temp_storage_bytes;
    int* temp_storage = NULL;
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, input, output, n);
    HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
    cudaDeviceSynchronize();
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, input, output, n);
    cudaDeviceSynchronize();
}

void ArrayMax(const float* input, float* max_val, size_t n) {
    size_t temp_storage_bytes;
    float* temp_storage = NULL;
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, input, max_val, n);
    HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
    cudaDeviceSynchronize();
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, input, max_val, n);
    cudaDeviceSynchronize();
}

void ArrayMin(const float* input, float* min_val, size_t n) {
    size_t temp_storage_bytes;
    float* temp_storage = NULL;
    cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, input, min_val, n);
    HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
    cudaDeviceSynchronize();
    cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, input, min_val, n);
    cudaDeviceSynchronize();
}

static __global__
void filterElemGt(const float* input, float* output, float th, size_t n) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadId >= n) return;
    if (input[threadId] > th) output[threadId] = -1.;
    else {
        output[threadId] = input[threadId];
    }
}

void ArrayArgmax(const float* input, float* max_val, int* max_idx, size_t n, float trunc) {
    float* filteredArray = nullptr;
    int BLOCK_SIZE = 32;
    int _grid = int(sqrt(double((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1))) + 1;
    dim3 gridSize(_grid, _grid, 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    if (trunc > 0.)
    {
        HANDLE_ERROR(cudaMalloc(&filteredArray, n * sizeof(float)));
        filterElemGt << <gridSize, dimBlockSize >> > (input, filteredArray, trunc, n);
    }
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    size_t temp_storage_bytes;
    int* temp_storage = NULL;
    cub::KeyValuePair<int, float>* h_out = new cub::KeyValuePair<int, float>;
    cub::KeyValuePair<int, float>* d_out;
    cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, float>));
    if (trunc > 0.)
    {
        cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, filteredArray, d_out, n);
        HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
        cudaDeviceSynchronize();
        cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, filteredArray, d_out, n);
        cudaDeviceSynchronize();
    }
    else
    {
        cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, input, d_out, n);
        HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
        cudaDeviceSynchronize();
        cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, input, d_out, n);
        cudaDeviceSynchronize();
    }
    HANDLE_ERROR(cudaMemcpy(max_idx, &(d_out->key), sizeof(int), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(max_val, &(d_out->value), sizeof(float), cudaMemcpyDeviceToDevice));
}

void ArrayArgmin(const float* input, float* min_val, int* min_idx, size_t n) {
    int BLOCK_SIZE = 32;
    int _grid = int(sqrt(double((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1))) + 1;
    dim3 gridSize(_grid, _grid, 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

    size_t temp_storage_bytes;
    float* temp_storage = NULL;
    cub::KeyValuePair<int, float>* h_out = new cub::KeyValuePair<int, float>;
    cub::KeyValuePair<int, float>* d_out;
    cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, float>));
    cub::DeviceReduce::ArgMin(temp_storage, temp_storage_bytes, input, d_out, n);
    HANDLE_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));
    cudaDeviceSynchronize();
    cub::DeviceReduce::ArgMin(temp_storage, temp_storage_bytes, input, d_out, n);
    cudaDeviceSynchronize();

    HANDLE_ERROR(cudaMemcpy(min_idx, &(d_out->key), sizeof(int), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(min_val, &(d_out->value), sizeof(float), cudaMemcpyDeviceToDevice));
}

static __global__
void filterElem2D(const float* input, float* output, int* valid, size_t n, size_t new_n) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadX >= new_n || threadY >= new_n) return;
    output[threadX * new_n + threadY] = input[valid[threadX] * n + valid[threadY]];
}

static __global__
void filterElem1D(const float* input, float* output, int* valid, size_t new_n) {
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId >= new_n) return;
    output[threadId] = input[valid[threadId]];
}

static __global__
void filterElem1DF3(const float3* input, float3* output, int* valid, size_t new_n) {
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId >= new_n) return;
    output[threadId] = input[valid[threadId]];
}

void filterArray(const float* input, float** output, int* valid, int ord, size_t n, size_t new_n) {
    if (ord == 1)
    {
        HANDLE_ERROR(cudaMalloc(output, sizeof(float) * new_n));
        int BLOCK_SIZE = 32;
        int _grid = (new_n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1;
        dim3 gridSize(_grid, 1, 1);
        dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
        filterElem1D << <gridSize, dimBlockSize >> > (input, *output, valid, new_n);
    }
    else if (ord == 2)
    {
        HANDLE_ERROR(cudaMalloc(output, sizeof(float) * new_n * new_n));
        int BLOCK_SIZE = 32;
        int _grid = (new_n - 1) / (BLOCK_SIZE) + 1;
        dim3 gridSize(_grid, _grid, 1);
        dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
        filterElem2D << <gridSize, dimBlockSize >> > (input, *output, valid, n, new_n);
    }
}

void filterArray3(const float3* input, float3** output, int* valid, size_t new_n) {
    HANDLE_ERROR(cudaMalloc(output, sizeof(float) * new_n * 3));
    int _grid = (new_n - 1) / (32 * 16) + 1;
    dim3 gridSize(_grid, 1, 1);
    dim3 dimBlockSize(32, 16, 1);
    filterElem1DF3 << <gridSize, dimBlockSize >> > (input, *output, valid, new_n);
}
