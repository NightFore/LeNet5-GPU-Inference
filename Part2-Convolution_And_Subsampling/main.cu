// Part2-LeNet5_Convolution_And_Subsampling.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>

#define SIZE_RAW_DATA 32
#define SIZE_C1_DATA 28
#define SIZE_S1_DATA 14
#define SIZE_KERNEL 5
#define NUM_KERNELS 6

// Create raw data and kernel matrices
float raw_data[SIZE_RAW_DATA * SIZE_RAW_DATA];  // 32x32 input image
float C1_data[NUM_KERNELS * SIZE_C1_DATA * SIZE_C1_DATA] = { 0 };  // 6x28x28 after convolution
float S1_data[NUM_KERNELS * SIZE_S1_DATA * SIZE_S1_DATA] = { 0 };  // 6x14x14 after subsampling
float C1_kernel[NUM_KERNELS * SIZE_KERNEL * SIZE_KERNEL];  // 6x5x5 kernels


/*
Helper CPU Methods
    - MatrixInit
    - MatrixPrint
    - TensorPrint
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

// Prints a tensor in a formatted manner.
void TensorPrint(float* T, int depth, int rows, int cols) {
    for (int d = 0; d < depth; d++) {
        printf("Depth %d:\n", d);
        MatrixPrint(&T[d * rows * cols], rows, cols);  // Call MatrixPrint to print the 2D slice of the tensor (at depth 'd')
        printf("\n");
    }
}


/*
Helper Kernels
    - applyConvolution
    - convolution2D_kernel
    - subsample2D_kernel
    - activation_kernel
*/

// Device kernel for convolution
__device__ float applyConvolution(float* input, float* kernel, int input_size, int kernel_size, int row, int col) {
    float result = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            result += input[(row + i) * input_size + (col + j)] * kernel[i * kernel_size + j];
        }
    }
    return result;
}

// Device function to apply tanh activation
__device__ float activation_tanh(float M) {
    return tanhf(M);
}

// Convolution kernel to apply to raw_data with kernels
__global__ void convolution2D_kernel(float* raw_data, float* C1_kernel, float* C1_data, int input_size, int kernel_size) {
    int k = blockIdx.z;  // Kernel index (depth)
    int i = blockIdx.y;  // Row in output matrix
    int j = threadIdx.x; // Column in output matrix

    if (i < SIZE_C1_DATA && j < SIZE_C1_DATA) {
        float conv_result = applyConvolution(raw_data, &C1_kernel[k * kernel_size * kernel_size], input_size, kernel_size, i, j);
        C1_data[k * SIZE_C1_DATA * SIZE_C1_DATA + i * SIZE_C1_DATA + j] = activation_tanh(conv_result);
    }
}

// Subsampling kernel to reduce the dimensions
__global__ void subsample2D_kernel(float* C1_data, float* S1_data, int input_size) {
    int k = blockIdx.z;  // Kernel index (depth)
    int i = blockIdx.y;  // Row in output matrix
    int j = threadIdx.x; // Column in output matrix

    if (i < SIZE_S1_DATA && j < SIZE_S1_DATA) {
        float sum = 0.0f;
        for (int m = 0; m < 2; m++) {
            for (int n = 0; n < 2; n++) {
                sum += C1_data[k * input_size * input_size + (2 * i + m) * input_size + (2 * j + n)];
            }
        }
        S1_data[k * SIZE_S1_DATA * SIZE_S1_DATA + i * SIZE_S1_DATA + j] = sum / 4.0f;
    }
}


/*
Main
    - main
*/

int main() {
    // Initialize random values for raw data and kernels
    MatrixInit(raw_data, SIZE_RAW_DATA, SIZE_RAW_DATA);
    MatrixInit(C1_kernel, NUM_KERNELS, SIZE_KERNEL * SIZE_KERNEL);

    printf("Raw data (input):\n");
    MatrixPrint(raw_data, SIZE_RAW_DATA, SIZE_RAW_DATA);

    printf("\nKernel data:\n");
    MatrixPrint(C1_kernel, NUM_KERNELS, SIZE_KERNEL * SIZE_KERNEL);

    // Allocate memory for GPU
    float *d_raw_data, *d_C1_kernel, *d_C1_data, *d_S1_data;
    cudaMalloc(&d_raw_data, SIZE_RAW_DATA * SIZE_RAW_DATA * sizeof(float));
    cudaMalloc(&d_C1_kernel, NUM_KERNELS * SIZE_KERNEL * SIZE_KERNEL * sizeof(float));
    cudaMalloc(&d_C1_data, NUM_KERNELS * SIZE_C1_DATA * SIZE_C1_DATA * sizeof(float));
    cudaMalloc(&d_S1_data, NUM_KERNELS * SIZE_S1_DATA * SIZE_S1_DATA * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_raw_data, raw_data, SIZE_RAW_DATA * SIZE_RAW_DATA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, NUM_KERNELS * SIZE_KERNEL * SIZE_KERNEL * sizeof(float), cudaMemcpyHostToDevice);

    // Launch convolution kernel with activation (tanh)
    dim3 blockSize(SIZE_C1_DATA, 1, 1);
    dim3 gridSize(SIZE_C1_DATA, SIZE_C1_DATA, NUM_KERNELS);
    convolution2D_kernel <<<gridSize, blockSize>>> (d_raw_data, d_C1_kernel, d_C1_data, SIZE_RAW_DATA, SIZE_KERNEL);
    cudaDeviceSynchronize();

    printf("\nAfter convolution (C1 data):\n");
    cudaMemcpy(C1_data, d_C1_data, NUM_KERNELS * SIZE_C1_DATA * SIZE_C1_DATA * sizeof(float), cudaMemcpyDeviceToHost);
    TensorPrint(C1_data, NUM_KERNELS, SIZE_C1_DATA, SIZE_C1_DATA);

    // Launch subsampling kernel
    subsample2D_kernel <<<gridSize, blockSize>>> (d_C1_data, d_S1_data, SIZE_C1_DATA);
    cudaDeviceSynchronize();

    printf("\nAfter subsampling (S1 data):\n");
    cudaMemcpy(S1_data, d_S1_data, NUM_KERNELS * SIZE_S1_DATA * SIZE_S1_DATA * sizeof(float), cudaMemcpyDeviceToHost);
    TensorPrint(S1_data, NUM_KERNELS, SIZE_S1_DATA, SIZE_S1_DATA);

    // Free device memory
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);

    return 0;
}
