// Part2-Convolution_And_Subsampling.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>

// Input Layer
#define INPUT_SIZE 32                                                       // Input image size (32x32)
float input[INPUT_SIZE * INPUT_SIZE];                                       // 32x32 input image

// Layer 1: Conv2D (C1)
#define C1_SIZE 28                                                          // Output size after first convolution (28x28)
#define C1_KERNEL_DEPTH 6                                                   // Number of kernels in the first convolution layer (depth)
#define C1_KERNEL_SIZE 5                                                    // Kernel size (5x5)
float C1_output[C1_KERNEL_DEPTH * C1_SIZE * C1_SIZE];                       // 6x28x28 output after convolution
float C1_weights[C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE];        // 6x5x5 kernel weights for the convolution
float C1_biases[C1_KERNEL_DEPTH];                                           // Bias for each kernel in C1

// Layer 2: Average Pooling (S2)
#define S2_SIZE 14                                                          // Output size after pooling (14x14)
float S2_output[C1_KERNEL_DEPTH * S2_SIZE * S2_SIZE];                       // 6x14x14 after average pooling


/*
Helper Methods
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
        // Call MatrixPrint to print the 2D slice of the tensor (at depth 'd')
        printf("Depth %d:\n", d);
        MatrixPrint(&T[d * rows * cols], rows, cols);
        printf("\n");
    }
}


/*
Helper Functions & Kernels
    - activation_tanh
    - convolution2D_kernel
    - subsample2D_kernel
*/

// Device function to apply tanh activation
__device__ float activation_tanh(float x) {
    return tanhf(x);
}

// Kernel function for performing 2D convolution with multiple kernels
__global__ void convolution2D_kernel(float* input, int input_size, float* output, int output_size, int kernel_size, float* weights, float* biases) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int depth = blockIdx.z;  // Depth (kernel index)

    // Ensure we're within the output dimensions
    if (x < output_size && y < output_size) {
        float sum = 0.0f;

        // Perform convolution operation
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_x = x + i;
                int input_y = y + j;

                // Check bounds to ensure we don't access outside the input
                if (input_x < input_size && input_y < input_size) {
                    int input_idx = input_y * input_size + input_x;                             // Flattened index of input
                    int kernel_idx = depth * kernel_size * kernel_size + i * kernel_size + j;   // Flattened index of kernel

                    // Multiply input value with the corresponding kernel and accumulate result
                    sum += input[input_idx] * weights[kernel_idx];
                }
            }
        }

        // Add bias for this kernel's output
        sum += biases[depth];

        // Apply activation function (tanh) to the result of convolution
        float activated_value = activation_tanh(sum);

        // Store the activated value in the output data
        int output_idx = depth * output_size * output_size + y * output_size + x;
        output[output_idx] = activated_value;
    }
}

// Kernel function for performing 2D subsampling (average pooling)
__global__ void subsample2D_kernel(float* input, int input_size, float* output, int output_size) {
    int k = blockIdx.z;  // Depth (kernel index), one feature map at a time
    int i = blockIdx.y;  // Row in output matrix
    int j = threadIdx.x; // Column in output matrix

    // Ensure we're within the output dimensions
    if (i < output_size && j < output_size) {
        float sum = 0.0f;

        // Perform subsampling operation (2x2 average pooling)
        for (int m = 0; m < 2; m++) {
            for (int n = 0; n < 2; n++) {
                int input_x = 2 * i + m;
                int input_y = 2 * j + n;

                // Check bounds to ensure we don't access outside the input
                if (input_x < input_size && input_y < input_size) {
                    int input_idx = k * input_size * input_size + input_y * input_size + input_x;   // Flattened index of input
                    sum += input[input_idx];                                                        // Accumulate values
                }
            }
        }

        // Store the result in the output data (taking average)
        int output_idx = k * output_size * output_size + i * output_size + j;   // Flattened index of output
        output[output_idx] = sum / 4.0f;                                        // Average of 2x2 block
    }
}

/*
Main
    - main
*/

int main() {
    // Initialize random values for input data and kernels
    MatrixInit(input, INPUT_SIZE, INPUT_SIZE);
    MatrixInit(C1_weights, C1_KERNEL_DEPTH, C1_KERNEL_SIZE * C1_KERNEL_SIZE);
    MatrixInit(C1_biases, C1_KERNEL_DEPTH, 1);

    // Allocate memory for GPU
    float *d_input;
    float *d_C1_weights, *d_C1_output, *d_C1_biases;
    float *d_S2_output;

    cudaMalloc(&d_input, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_C1_weights, C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C1_output, C1_KERNEL_DEPTH * C1_SIZE * C1_SIZE * sizeof(float));
    cudaMalloc(&d_C1_biases, C1_KERNEL_DEPTH * sizeof(float));
    cudaMalloc(&d_S2_output, C1_KERNEL_DEPTH * S2_SIZE * S2_SIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_weights, C1_weights, C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_biases, C1_biases, C1_KERNEL_DEPTH * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Apply convolution (C1)
    dim3 blockSize(C1_SIZE, 1, 1);
    dim3 gridSize(C1_SIZE, C1_SIZE, C1_KERNEL_DEPTH);
    convolution2D_kernel << <gridSize, blockSize >> > (d_input, INPUT_SIZE, d_C1_output, C1_SIZE, C1_KERNEL_SIZE, d_C1_weights, d_C1_biases);
    cudaDeviceSynchronize();

    // Step 2: Apply subsampling (S2)
    dim3 blockSizeS2(S2_SIZE, 1, 1);
    dim3 gridSizeS2(S2_SIZE, S2_SIZE, C1_KERNEL_DEPTH);
    subsample2D_kernel << <gridSize, blockSize >> > (d_C1_output, C1_SIZE, d_S2_output, S2_SIZE);
    cudaDeviceSynchronize();

    // Results
    printf("Input data:\n");
    MatrixPrint(input, INPUT_SIZE, INPUT_SIZE);

    printf("\nKernel data:\n");
    MatrixPrint(C1_weights, C1_KERNEL_DEPTH, C1_KERNEL_SIZE * C1_KERNEL_SIZE);

    printf("\nKernel biases (C1):\n");
    MatrixPrint(C1_biases, C1_KERNEL_DEPTH, 1);

    printf("\nAfter convolution (C1 data):\n");
    cudaMemcpy(C1_output, d_C1_output, C1_KERNEL_DEPTH * C1_SIZE * C1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    TensorPrint(C1_output, C1_KERNEL_DEPTH, C1_SIZE, C1_SIZE);

    printf("\nAfter subsampling (S2 data):\n");
    cudaMemcpy(S2_output, d_S2_output, C1_KERNEL_DEPTH * S2_SIZE * S2_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    TensorPrint(S2_output, C1_KERNEL_DEPTH, S2_SIZE, S2_SIZE);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_C1_weights);
    cudaFree(d_C1_output);
    cudaFree(d_C1_biases);
    cudaFree(d_S2_output);

    return 0;
}
