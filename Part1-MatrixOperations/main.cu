// Part1-MatrixOperations.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "device_launch_parameters.h"

// Flag to print matrices
bool printMatrices = false;

// Matrix dimensions and configuration parameters
int n = 1000, p = 1000;

// Memory pointers for host and device
float* matrix1, * matrix2, * resultMatrix;
float* d_M1, * d_M2, * d_Mout;

// Timing variables for performance measurements
clock_t cpuStart, cpuEnd;
cudaEvent_t gpuStart, gpuStop;
float cpuTime, gpuTime;


/*
Helper Methods
    - MatrixInit
    - MatrixPrint
    - MatrixAdd
    - MatrixMult
*/

// Initializes a matrix with random values between -1 and 1.
void MatrixInit(float* M, int n, int p) {
    // Iterate through each element of the matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Assign a random value between -1 and 1 to each element
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

// Prints a matrix in a formatted manner.
void MatrixPrint(float* M, int n, int p) {
    // Iterate through each element of the matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Print each element of the matrix with 2 decimal places
            printf("%6.2f ", M[i * p + j]);
        }
        printf("\n");
    }
}

// CPU method to performs element-wise addition of two matrices.
void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
    // Iterate through each element of the matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Perform the matrix addition
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// CPU method to perform matrix multiplication for two square matrices.
void MatrixMult(float* M1, float* M2, float* Mout, int n) {
    // Iterate through each element of the matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Initialize the output element to 0
            Mout[i * n + j] = 0;

            // Perform the matrix multiplication
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}


/*
Helper Kernels
    - cudaMatrixAdd
    - cudaMatrixMult
*/

// CUDA kernel to perform element-wise addition of two matrices.
__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
    // Calculate the row and column indices
    int row = blockIdx.x;
    int col = threadIdx.x;

    // Ensure we don't go out of bounds
    if (row < n && col < p) {
        // Perform the matrix addition
        Mout[row * p + col] = M1[row * p + col] + M2[row * p + col];
    }
}

// CUDA kernel to perform matrix multiplication for two square matrices.
__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n) {
    // Calculate the row and column indices
    int row = blockIdx.x;
    int col = threadIdx.x;

    // Ensure we don't go out of bounds
    if (row < n && col < n) {
        // Initialize the output element to 0
        Mout[row * n + col] = 0.0f;

        // Perform the matrix multiplication
        for (int k = 0; k < n; k++) {
            Mout[row * n + col] += M1[row * n + k] * M2[k * n + col];
        }
    }
}


/*
Main
    - main
*/
int main(int argc, char *argv[]) {
    // Override default values based on command line arguments (if provided)
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) p = atoi(argv[2]);
    if (argc > 3 && argv[3][0] == '1') printMatrices = true;

    // Allocate memory for matrices
    float *matrix1 = (float*) malloc(n * p * sizeof(float));
    float *matrix2 = (float*) malloc(n * p * sizeof(float));
    float *resultMatrix = (float*) malloc(n * p * sizeof(float));

    // Timing variables
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);

    printf("---------- Matrix Initialization ----------\n");
    // Initialize matrices with random values
    MatrixInit(matrix1, n, p);
    printf("Matrix 1 initialized\n");

    MatrixInit(matrix2, n, p);
    printf("Matrix 2 initialized\n");

    // Print the result
    if (printMatrices) {
        MatrixPrint(matrix1, n, p);
        MatrixPrint(matrix2, n, p);
    }

    printf("\n---------- Matrix Addition (CPU) ----------\n");
    cpuStart = clock();
    MatrixAdd(matrix1, matrix2, resultMatrix, n, p);
    cpuEnd = clock();
    cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
    printf("Time taken for CPU Addition: %f seconds\n", cpuTime);

    printf("\n---------- Matrix Addition (GPU) ----------\n");
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_M1, matrix1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, matrix2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel and measure the time
    cudaEventRecord(gpuStart);
    cudaMatrixAdd<<<n, p>>>(d_M1, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();
    cudaEventRecord(gpuStop);
    cudaEventSynchronize(gpuStop);
    cudaEventElapsedTime(&gpuTime, gpuStart, gpuStop);

    // Copy the result from device to host
    cudaMemcpy(resultMatrix, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    // Print the result
    printf("Time taken for GPU Addition: %f seconds\n", gpuTime / 1000);
    if (printMatrices) {
        printf("\nResult of Matrix Addition:\n");
    	MatrixPrint(resultMatrix, n, p);
    }

    printf("\n---------- Matrix Multiplication (CPU) ----------\n");
    cpuStart = clock();
    MatrixMult(matrix1, matrix2, resultMatrix, n);
    cpuEnd = clock();
    cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
    printf("Time taken for CPU Multiplication: %f seconds\n", cpuTime);

    printf("\n---------- Multiplication Part (GPU) ----------\n");
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_M1, matrix1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, matrix2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel and measure the time
    cudaEventRecord(gpuStart);
    cudaMatrixMult <<<n, n>>> (d_M1, d_M2, d_Mout, n);
    cudaDeviceSynchronize();
    cudaEventRecord(gpuStop);
    cudaEventSynchronize(gpuStop);
    cudaEventElapsedTime(&gpuTime, gpuStart, gpuStop);

    // Copy the result from device to host
    cudaMemcpy(resultMatrix, d_Mout, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Time taken for GPU Multiplication: %f seconds\n", gpuTime / 1000);
    if (printMatrices) {
        printf("\nResult of Matrix Multiplication:\n");
        MatrixPrint(resultMatrix, n, n);
    }

    // Free device memory
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    // Free allocated memory
    free(matrix1);
    free(matrix2);
    free(resultMatrix);

    return 0;
}
