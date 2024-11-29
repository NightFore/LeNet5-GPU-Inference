#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Function prototypes
void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
void MatrixMult(float *M1, float *M2, float *Mout, int n);

// CUDA kernel to add two matrices
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
{
    int row = blockIdx.x;  // blockIdx.x corresponds to the row (block)
    int col = threadIdx.x; // threadIdx.x corresponds to the column (thread)

    if (row < n && col < p) {
        Mout[row * p + col] = M1[row * p + col] + M2[row * p + col];
    }
}

// CUDA kernel to multiply two NxN matrices
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n)
{
    int row = blockIdx.x;  // blockIdx.x corresponds to the row (block)
    int col = threadIdx.x; // threadIdx.x corresponds to the column (thread)

    if (row < n && col < n) {
        float value = 0.0f;
        for (int k = 0; k < n; k++) {
            value += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = value;
    }
}

int main(int argc, char *argv[])
{
    // Size of the matrices
    int n = atoi(argv[1]);  // Change this as needed (size of matrix)
    int p = atoi(argv[2]);  // Change this as needed (size of matrix)

    // Declare matrices
    float *matrix1 = (float*)malloc(n * p * sizeof(float));
    float *matrix2 = (float*)malloc(n * p * sizeof(float));
    float *resultMatrix = (float*)malloc(n * p * sizeof(float));

    // Initialize matrices with random values
    MatrixInit(matrix1, n, p);
    MatrixInit(matrix2, n, p);

    printf("Matrix 1 initialized:\n");
    MatrixPrint(matrix1, n, p);
    printf("\nMatrix 2 initialized:\n");
    MatrixPrint(matrix2, n, p);

    // Measure time for CPU addition
    clock_t cpuStart, cpuEnd;
    double cpuTime;

    // ********************** Matrix Addition (CPU) ***********************
    printf("\n**************** Addition Part (CPU) ****************\n");
    cpuStart = clock();
    MatrixAdd(matrix1, matrix2, resultMatrix, n, p);
    cpuEnd = clock();
    cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
    printf("Time taken for CPU Addition: %f seconds\n", cpuTime);

    // ********************** Matrix Addition (GPU) ***********************
    printf("\n**************** Addition Part (GPU) ****************\n");
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    cudaMemcpy(d_M1, matrix1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, matrix2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Timing GPU operation
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMatrixAdd<<<n, p>>>(d_M1, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Time taken for GPU Addition: %f ms\n", elapsedTime);

    // Copy result back from device
    cudaMemcpy(resultMatrix, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    if (argv[3][0] == '1') {
        printf("\nResult of Matrix Addition:\n");
    	MatrixPrint(resultMatrix, n, p);
    }
    

    // ********************** Matrix Multiplication (CPU) ***********************
    printf("\n**************** Multiplication Part (CPU) ****************\n");
    cpuStart = clock();
    MatrixMult(matrix1, matrix2, resultMatrix, n);
    cpuEnd = clock();
    cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
    printf("Time taken for CPU Multiplication: %f seconds\n", cpuTime);

    // ********************** Matrix Multiplication (GPU) ***********************
    printf("\n**************** Multiplication Part (GPU) ****************\n");

    cudaMalloc((void**)&d_M1, n * n * sizeof(float));
    cudaMalloc((void**)&d_M2, n * n * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * n * sizeof(float));

    cudaMemcpy(d_M1, matrix1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, matrix2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    cudaMatrixMult<<<n, n>>>(d_M1, d_M2, d_Mout, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time taken for GPU Multiplication: %f ms\n", elapsedTime);

    // Copy result back from device
    cudaMemcpy(resultMatrix, d_Mout, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
    
    if (argv[3][0] == '1') {
        printf("\nResult of Matrix Multiplication:\n");
    	MatrixPrint(resultMatrix, n, n);
    }

    // Free the allocated memory
    free(matrix1);
    free(matrix2);
    free(resultMatrix);

    return 0;
}

// Function to initialize the matrix with random values
void MatrixInit(float *M, int n, int p)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random value between -1 and 1
        }
    }
}

// Function to print the matrix
void MatrixPrint(float *M, int n, int p)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%6.2f ", M[i * p + j]);
        }
        printf("\n");
    }
}

// Function to add two matrices M1 and M2 of size n x p, storing the result in Mout (CPU version)
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// Function to multiply two matrices M1 and M2 of size n x p, storing the result in Mout (CPU version)
void MatrixMult(float *M1, float *M2, float *Mout, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

