
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>

#define S1 "jak"

#define S2 "tak"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__host__ uint16_t checkWord(char* s);

__host__ __device__ uint32_t GetDInd(uint16_t i, uint16_t j, uint16_t width);

__host__ __device__ uint16_t Min(uint16_t num1, uint16_t num2, uint16_t num3);

__host__ void CPULevenshtein(char* s1, uint16_t s1Len, char* s2, uint16_t s2Len, uint16_t* D);

__host__ void PrintD(uint16_t* D, uint16_t height, uint16_t width);

__global__ void addKernel(int *c, const int *a, const int *b)
{

}

int main(int argc, char* argv[])
{
    cudaError_t cudaStatus;
    
    char* s1, *s2;

    if (argc > 2)
        s2 = argv[2];
    else
        s2 = S2;
    if (argc > 1)
        s1 = argv[1];
    else
    {
        s1 = S1; s2 = S2;
    }

    uint16_t s1Len = checkWord(s1), s2Len = checkWord(s2);
    if (s1Len == 0 || s2Len == 0)
    {
        printf("Podane slowa sa niepoprawne!\n");
        return 0;
    }

    uint16_t* D_CPU = (uint16_t*)malloc(sizeof(uint16_t) * (s1Len + 1) * (s2Len + 1));

    CPULevenshtein(s1, s1Len, s2, s2Len, D_CPU);

    PrintD(D_CPU, s1Len + 1, s2Len + 1);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    free(D_CPU);
    return 0;
}

__host__ uint16_t checkWord(char* s)
{
    int i = 0;
    while (s[i] != '\0')
    {
        if (s[i] < 97 || s[i] > 122)
            return 0;
        i++;
    }
    return i;
}

__host__ __device__ uint32_t GetDInd(uint16_t i, uint16_t j, uint16_t width)
{
    return i * width + j;
}

__host__ __device__ uint16_t Min(uint16_t num1, uint16_t num2, uint16_t num3)
{
    if (num1 <= num2 && num1 <= num3)
        return num1;
    else if (num2 <= num1 && num2 <= num3)
        return num2;
    else return num3;
}

__host__ void CPULevenshtein(char* s1, uint16_t s1Len, char* s2, uint16_t s2Len, uint16_t* D)
{
    for (uint16_t i = 0; i < s1Len + 1; i++)
    {
        for (uint16_t j = 0; j < s2Len + 1; j++)
        {
            if (i == 0)
                D[GetDInd(i, j, s2Len + 1)] = j;
            else if (j == 0)
                D[GetDInd(i, j, s2Len + 1)] = i;
            else if (s1[i - 1] == s2[j - 1])
                D[GetDInd(i, j, s2Len + 1)] = D[GetDInd(i - 1, j - 1, s2Len + 1)];
            else
                D[GetDInd(i, j, s2Len + 1)] = 1 + Min(D[GetDInd(i - 1, j, s2Len + 1)], D[GetDInd(i, j - 1, s2Len + 1)], D[GetDInd(i - 1, j - 1, s2Len + 1)]);
        }
    }
}

__host__ void PrintD(uint16_t* D, uint16_t height, uint16_t width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%3d", D[GetDInd(i, j, width)]);
        }
        printf("\n");
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}