#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

static const char* cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";

        default:
            return "<unknown>";
    }
}

inline void cublasLtCheck(cublasStatus_t status, int iLine) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CublasLt error " << cublasGetErrorEnum(status) << " at line " << iLine << endl;
    }
}

#define cublasLtCk(call) cublasLtCheck(call, __LINE__)

float MatrixMultiply(int m, int k, int n);

void ComputeDim(string fileName)
{
    int m, k, n;
    ifstream inFile(fileName, ios::in);
    string lineStr;
    vector<vector<int> > intArray;
    getline(inFile, lineStr);
    while (getline(inFile, lineStr))
    {
        cout << lineStr << endl;
        stringstream ss(lineStr);
        string str;
        vector<int> lineArray;
        
        while (getline(ss, str, ','))
        {
            lineArray.push_back(stoi(str));
        }
        intArray.push_back(lineArray);
    }

    for (int i = 0; i < intArray.size(); ++i) {
        vector<int> line = intArray[i];
        for (int bs=1; bs <= 32; bs*=2) {
            m = bs * line[4] * line[5];
            k = line[1] * line[2] * line[3];
            n = line[0];
            cout << "convID:" << i << " bs=" << bs << " m=" << m << " k=" << k << " n=" << n << " time=" << MatrixMultiply(m, k, n) << endl;
        }
    }
    return;
}


void ConstantInit(float* data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

float MatrixMultiply(int m, int k, int n)
{
    float computingTime = -1;
    // Create cublas handle
    cublasHandle_t handle;
    cublasLtCk(cublasCreate(&handle));
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    
    // Allocate host memory for matrices A and B
    unsigned int size_A = m * k;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));

    unsigned int size_B = k * n;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

    // Initialize host memory
    const float valB = 0.1f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate host matrix C
    unsigned int mem_size_C = m * n * sizeof(float);
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A);
    cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B);
    cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C);

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream);
    
    // Setup parameters
    // C = a(AB) + bC
    float a = 1.0f;
    float b = 0.0f;

    // Performs warmup operation
    cublasLtCk(cublasGemmEx(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &a,
            d_A,
            CUDA_R_32F,
            m,
            d_B,
            CUDA_R_32F,
            k,
            &b,
            d_C,
            CUDA_R_32F,
            m,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_AlGO0
        ));
    cudaStreamSynchronize(stream);
    // Record the start event
    cudaEventRecord(start, stream);
    int nIter = 20;
    for (int i = 0; i < nIter; ++i) {
        cublasLtCk(cublasGemmEx(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &a,
                d_A,
                CUDA_R_32F,
                m,
                d_B,
                CUDA_R_32F,
                k,
                &b,
                d_C,
                CUDA_R_32F,
                m,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_ALGO0
            ));
    }
    // Record the stop event
    cudaEventRecord(stop, stream);
    
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    float msecPerMatrixMul = msecTotal / nIter;
    computingTime = msecPerMatrixMul * 1000;

    // Print the result matrix
      
    // std::cout << "Time: " << msec << " ms" << std::endl; 

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return computingTime; 
    
}

int main()
{
    ComputeDim("resnet50_conv.csv");
    // cout << MatrixMultiply(64, 100, 100) << endl;
    // cout << MatrixMultiply(128, 100, 100) << endl;
}

