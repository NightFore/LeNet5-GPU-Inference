﻿// Part3-LeNet5.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>

// Input Layer
#define INPUT_SIZE 32                                                       // Input image size (32x32)
float input[INPUT_SIZE * INPUT_SIZE];                                       // Normalized pixel data

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

// Layer 3: Conv2D (C3)
#define C3_SIZE 10                                                          // Output size after second convolution (10x10)
#define C3_KERNEL_DEPTH 16                                                  // Number of kernels in second convolution layer (depth)
#define C3_KERNEL_SIZE 5                                                    // Kernel size (5x5) for C3
float C3_output[C3_KERNEL_DEPTH * C3_SIZE * C3_SIZE];                       // 16x10x10 output after second convolution
float C3_weights[C3_KERNEL_DEPTH * C3_KERNEL_SIZE * C3_KERNEL_SIZE];        // 16x5x5 kernel weights for the second convolution
float C3_biases[C3_KERNEL_DEPTH];                                           // Bias for each kernel in C3

// Layer 4: Average Pooling (S4)
#define S4_SIZE 5                                                           // Output size after pooling (5x5)
float S4_output[C3_KERNEL_DEPTH * S4_SIZE * S4_SIZE];                       // 16x5x5 after average pooling

// Flatten (Converting 2D data into 1D)
#define FLATTEN_SIZE (S4_SIZE * S4_SIZE * C3_KERNEL_DEPTH)                  // Flattened size before fully connected
float flattened_data[FLATTEN_SIZE];                                         // Flattened data output from S4

// Layer 5: Fully Connected (F5)
#define F5_SIZE 120                                                         // Number of neurons in F5
float F5_output[F5_SIZE];                                                   // Output from F5
float F5_weights[F5_SIZE * (S4_SIZE * S4_SIZE * C3_KERNEL_DEPTH)];          // Weights for F5
float F5_biases[F5_SIZE];                                                   // Bias for each neuron in F5

// Layer 6: Fully Connected (F6)
#define F6_SIZE 84                                                          // Number of neurons in F6
float F6_output[F6_SIZE];                                                   // Output from F6
float F6_weights[F6_SIZE * F5_SIZE];                                        // Weights for F6
float F6_biases[F6_SIZE];                                                   // Bias for each neuron in F6

// Layer 7: Fully Connected (F7: Output)
#define F7_SIZE 10                                                          // Number of output classes (10 for MNIST)
float F7_output[F7_SIZE];                                                   // Output from F7
float F7_weights[F7_SIZE * F6_SIZE];                                        // Weights for F7
float F7_biases[F7_SIZE];                                                   // Bias for each neuron in F7

// Placeholder
#define WIDTH 28
#define HEIGHT 28
#define FILENAME "data/train-images.idx3-ubyte"
int imgIndex = 1;
int*** img;                                                                 // Image data
int color[3] = { 255, 0, 0 };                                               // RGB color for visualizing the images

/*
Helper Methods
    - MatrixInit
    - MatrixPrint
    - TensorPrint
    - charBackgroundPrint
    - imgColorPrint
    - create_and_print_image
    - createGrayscaleImage
*/
// Initializes a matrix with random values between -1 and 1
void MatrixInit(float* M, int n, int p) {
    // Iterate through each element of the matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Assign a random value between -1 and 1 to each element
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

// Prints a matrix in a formatted manner
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

// Prints a tensor in a formatted manner
void TensorPrint(float* T, int depth, int rows, int cols) {
    for (int d = 0; d < depth; d++) {
        // Call MatrixPrint to print the 2D slice of the tensor (at depth 'd')
        printf("Depth %d:\n", d);
        MatrixPrint(&T[d * rows * cols], rows, cols);
        printf("\n");
    }
}

// Prints a string with a specified RGB background color
void charBackgroundPrint(char* str, int rgb[3]) {
    // Set background color using ANSI escape codes
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);

    // Print the string with the set background color
    printf("%s", str);

    // Reset terminal formatting to default (no background color)
    printf("\033[0m");
}

// Prints an image represented as a 3D array of RGB values
void imgColorPrint(int height, int width, int*** img) {
    int row, col;

    // Two spaces ("  ") to create a block-like effect for each pixel when printed.
    char* str = "  ";

    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            // Print the pixel with its corresponding RGB background color
            charBackgroundPrint(str, img[row][col]);
        }
        printf("\n");
    }
}

// Allocate memory, convert input to RGB, and print the image
void create_and_print_image(int height, int width) {
    // Allocate memory for the image
    img = (int***)malloc(sizeof(int**) * height);
    for (int i = 0; i < height; i++) {
        img[i] = (int**)malloc(sizeof(int*) * width);
        for (int j = 0; j < width; j++) {
            // RGB data storage
            img[i][j] = (int*)malloc(sizeof(int) * 3);
        }
    }

    // Convert the input data to RGB and store in img array
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Convert the pixel value in input to RGB for visualization
            int pixelVal = (int)(input[i * width + j] * 255);
            img[i][j][0] = (int)(pixelVal * color[0] / 255);  // Red channel
            img[i][j][1] = (int)(pixelVal * color[1] / 255);  // Green channel
            img[i][j][2] = (int)(pixelVal * color[2] / 255);  // Blue channel
        }
    }

    // Display the image
    imgColorPrint(height, width, img);

    // Free allocated memory after usage
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            free(img[i][j]);
        }
        free(img[i]);
    }
    free(img);
}


// Applies grayscale transformation to the image
void createGrayscaleImage(int height, int width, int*** img) {
    int i, j;

    // Apply grayscale transformation
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            img[i][j][0] = ((i + j) * 4) % 255;  // Apply grayscale pattern (R channel)
            img[i][j][1] = ((i + j) * 4) % 255;  // Apply grayscale pattern (G channel)
            img[i][j][2] = ((i + j) * 4) % 255;  // Apply grayscale pattern (B channel)
        }
    }

    // Print the resulting image
    imgColorPrint(height, width, img);
}


/*
Helper Functions & Kernels
    - tanh_kernel
    - softmax_kernel
    - convolution2D_kernel
    - subsample2D_kernel
    - flatten_kernel
    - fully_connected_kernel
*/
// Kernel to apply tanh activation to the fully connected layer output
__global__ void tanh_kernel(float* output, int output_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < output_size) {
        // Apply tanh activation function element-wise
        output[idx] = tanhf(output[idx]);
    }
}

// Kernel to apply softmax activation to the output layer
__global__ void softmax_kernel(float* output, int output_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < output_size) {
        // Step 1: Find the maximum value in the output array
        float max_val = output[0];
        for (int i = 1; i < output_size; i++) {
            if (output[i] > max_val) {
                max_val = output[i];
            }
        }

        // Step 2: Exponentiate each value minus the max value for numerical stability
        float exp_value = expf(output[idx] - max_val);

        // Step 3: Compute sum of exponentials in one thread (the sum is computed only once)
        float sum_exp = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum_exp += expf(output[i] - max_val);  // Avoid overflow using the max value
        }

        // Step 4: Normalize the exponentials to get the final softmax
        output[idx] = exp_value / sum_exp;
    }
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
        float activated_value = tanhf(sum);

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

// Kernel function for flattening 3D data into a 1D array
__global__ void flatten_kernel(float* input, int input_depth, int input_size, float* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_size = input_depth * input_size * input_size;

    if (idx < total_size) {
        int z = idx / (input_size * input_size);
        int y = (idx % (input_size * input_size)) / input_size;
        int x = idx % input_size;
        output[idx] = input[(z * input_size + y) * input_size + x];
    }
}

// Kernel function for fully connected layer
__global__ void fully_connected_kernel(float* input, int input_size, float* output, int output_size, float* weights, float* biases) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < output_size) {
        float sum = 0.0f;

        // Perform dot product of input and weights
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }

        // Add bias
        sum += biases[idx];

        // Store the result in output
        output[idx] = sum;
    }
}


/*
Main
    - initialize_input
    - initialize_weights
    - load_image
    - pad_image
    - load_weights
    - load_all_weights
    - run_lenet_gpu
    - main
*/
// Initialize input data with random values
void initialize_input() {
    MatrixInit(input, INPUT_SIZE, INPUT_SIZE);
}

// Initialize weights with random values
void initialize_weights() {
    MatrixInit(C1_weights, C1_KERNEL_DEPTH, C1_KERNEL_SIZE * C1_KERNEL_SIZE);
    MatrixInit(C1_biases, C1_KERNEL_DEPTH, 1);
    MatrixInit(C3_weights, C3_KERNEL_DEPTH, C3_KERNEL_SIZE * C3_KERNEL_SIZE);
    MatrixInit(C3_biases, C3_KERNEL_DEPTH, 1);
    MatrixInit(F5_weights, F5_SIZE, FLATTEN_SIZE);
    MatrixInit(F5_biases, F5_SIZE, 1);
    MatrixInit(F6_weights, F6_SIZE, F5_SIZE);
    MatrixInit(F6_biases, F6_SIZE, 1);
    MatrixInit(F7_weights, F7_SIZE, F6_SIZE);
    MatrixInit(F7_biases, F7_SIZE, 1);
}

// Load and preprocess images
int load_image(char* filename, int imgIndex, int width, int height) {
    // Initialize variables
    unsigned int magic, nbImg, nbRows, nbCols;  // Metadata
    unsigned char val;                          // Temporary variable to hold pixel data
    FILE* fptr;                                 // Pointer to the file to read from

    // Open file
    if ((fptr = fopen(filename, "rb")) == NULL) {
        printf("Error: Can't open file\n");
        return -1;
    }

    // Read the header
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    // Print the metadata values
    printf("Nb Magic : %u \n", magic);
    printf("Nb Img : %u \n", nbImg);
    printf("Nb Rows : %u \n", nbRows);
    printf("Nb Cols : %u \n", nbCols);

    // Skip over previous images if imgIndex > 1
    for (int skip = 0; skip < (imgIndex - 1) * height * width; skip++) {
        fread(&val, sizeof(unsigned char), 1, fptr);
    }

    // Read the image data
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fread(&val, sizeof(unsigned char), 1, fptr);
            input[i * width + j] = val / 255.0f;  // Normalize to range [0, 1]
        }
    }

    // Allocate memory for RGB img and print the "before padding" image
    img = (int***)malloc(sizeof(int**) * height);   // Using original height/width for the 3D array
    for (int i = 0; i < height; i++) {
        img[i] = (int**)malloc(sizeof(int*) * width);  // Using original width for the 3D array
        for (int j = 0; j < width; j++) {
            img[i][j] = (int*)malloc(sizeof(int) * 3);   // RGB channels
        }
    }

    // Convert the input data to RGB and store it in the img array for "before padding"
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixelVal = (int)(input[i * width + j] * 255); // RGB conversion for visualization
            img[i][j][0] = (int)(pixelVal * color[0] / 255);  // Red channel
            img[i][j][1] = (int)(pixelVal * color[1] / 255);  // Green channel
            img[i][j][2] = (int)(pixelVal * color[2] / 255);  // Blue channel
        }
    }
 
    // Close file
    fclose(fptr);
}

// Pad the input image to match INPUT_SIZE x INPUT_SIZE
void pad_image(int height, int width) {
    int paddedWidth = INPUT_SIZE;
    int paddedHeight = INPUT_SIZE;

    // Initialize the padded image to zero
    float paddedImg[INPUT_SIZE][INPUT_SIZE];
    for (int i = 0; i < paddedHeight; i++) {
        for (int j = 0; j < paddedWidth; j++) {
            paddedImg[i][j] = 0.0f;  // Set default zero value
        }
    }

    // Offset to place the original image in the center of the padded image
    int offsetX = (paddedWidth - width) / 2;
    int offsetY = (paddedHeight - height) / 2;

    // Copy the image into the center of the padded image
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            paddedImg[i + offsetY][j + offsetX] = input[i * width + j];
        }
    }

    // Copy back the padded image
    for (int i = 0; i < paddedHeight; i++) {
        for (int j = 0; j < paddedWidth; j++) {
            input[i * paddedWidth + j] = paddedImg[i][j];
        }
    }
}

// Load weights from a file
void load_weights(const char* filename, float* weights, int size) {
    // Open the file in read mode
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s.\n", filename);
        exit(EXIT_FAILURE); // Exit if file opening fails
    }

    // Read weights from the file
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &weights[i]) != 1) {
            printf("Error: File format incorrect or insufficient data in %s.\n", filename);
            fclose(file);
            exit(EXIT_FAILURE); // Exit if reading fails
        }
    }

    // Close the file
    fclose(file);
    printf("Weights successfully loaded from %s.\n", filename);
}

// Load weights for all layers
void load_all_weights() {
    // File names for each layer's weight data
    const char* C1_weights_file = "data/C1_weights.txt";
    const char* C3_weights_file = "data/C3_weights.txt";
    const char* F5_weights_file = "data/F5_weights.txt";
    const char* F6_weights_file = "data/F6_weights.txt";
    const char* F7_weights_file = "data/F7_weights.txt";

    // Load weights for each layer
    load_weights(C1_weights_file, C1_weights, C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE);
    load_weights(C3_weights_file, C3_weights, C3_KERNEL_DEPTH * C3_KERNEL_SIZE * C3_KERNEL_SIZE);
    load_weights(F5_weights_file, F5_weights, F5_SIZE * (S4_SIZE * S4_SIZE * C3_KERNEL_DEPTH));
    load_weights(F6_weights_file, F6_weights, F6_SIZE * F5_SIZE);
    load_weights(F7_weights_file, F7_weights, F7_SIZE * F6_SIZE);

    // Print loaded weights for C1 (for debugging purposes)
    printf("\nC1 Kernel Weights:\n");
    for (int i = 0; i < C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE; i++) {
        printf("%f ", C1_weights[i]);
        if ((i + 1) % C1_KERNEL_SIZE == 0) {
            printf("\n");
        }
    }
}

// LeNet GPU function to allocate memory, run the network, and free memory
void run_lenet_gpu() {
    // Allocate memory for GPU
    float* d_input;
    float* d_C1_weights, *d_C1_output, *d_C1_biases;
    float* d_S2_output;
    float* d_C3_weights, *d_C3_output, *d_C3_biases;
    float* d_S4_output;
    float* d_flattened_data;
    float* d_F5_weights, *d_F5_output, *d_F5_biases;
    float* d_F6_weights, *d_F6_output, *d_F6_biases;
    float* d_F7_weights, *d_F7_output, *d_F7_biases;

    // CUDA memory allocation
    cudaMalloc(&d_input, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_C1_weights, C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C1_output, C1_KERNEL_DEPTH * C1_SIZE * C1_SIZE * sizeof(float));
    cudaMalloc(&d_C1_biases, C1_KERNEL_DEPTH * sizeof(float));
    cudaMalloc(&d_S2_output, C1_KERNEL_DEPTH * S2_SIZE * S2_SIZE * sizeof(float));
    cudaMalloc(&d_C3_weights, C3_KERNEL_DEPTH * C3_KERNEL_SIZE * C3_KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C3_output, C3_KERNEL_DEPTH * C3_SIZE * C3_SIZE * sizeof(float));
    cudaMalloc(&d_C3_biases, C3_KERNEL_DEPTH * sizeof(float));
    cudaMalloc(&d_S4_output, C3_KERNEL_DEPTH * S4_SIZE * S4_SIZE * sizeof(float));
    cudaMalloc(&d_flattened_data, C3_KERNEL_DEPTH * S4_SIZE * S4_SIZE * sizeof(float));
    cudaMalloc(&d_F5_weights, F5_SIZE * FLATTEN_SIZE * sizeof(float));
    cudaMalloc(&d_F5_output, F5_SIZE * sizeof(float));
    cudaMalloc(&d_F5_biases, F5_SIZE * sizeof(float));
    cudaMalloc(&d_F6_weights, F6_SIZE * F5_SIZE * sizeof(float));
    cudaMalloc(&d_F6_output, F6_SIZE * sizeof(float));
    cudaMalloc(&d_F6_biases, F6_SIZE * sizeof(float));
    cudaMalloc(&d_F7_weights, F7_SIZE * F6_SIZE * sizeof(float));
    cudaMalloc(&d_F7_output, F7_SIZE * sizeof(float));
    cudaMalloc(&d_F7_biases, F7_SIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_weights, C1_weights, C1_KERNEL_DEPTH * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_biases, C1_biases, C1_KERNEL_DEPTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_weights, C3_weights, C3_KERNEL_DEPTH * C3_KERNEL_SIZE * C3_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_biases, C3_biases, C3_KERNEL_DEPTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flattened_data, d_S4_output, C3_KERNEL_DEPTH * S4_SIZE * S4_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_F5_weights, F5_weights, F5_SIZE * FLATTEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F5_biases, F5_biases, F5_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F6_weights, F6_weights, F6_SIZE * F5_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F6_biases, F6_biases, F6_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F7_weights, F7_weights, F7_SIZE * F6_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F7_biases, F7_biases, F7_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Apply convolution (C1)
    dim3 blockSize(C1_SIZE, 1, 1);
    dim3 gridSize(C1_SIZE, C1_SIZE, C1_KERNEL_DEPTH);
    convolution2D_kernel << <gridSize, blockSize >> > (d_input, INPUT_SIZE, d_C1_output, C1_SIZE, C1_KERNEL_SIZE, d_C1_weights, d_C1_biases);
    cudaDeviceSynchronize();

    // Step 2: Apply subsampling (S2)
    dim3 blockSizeS2(S2_SIZE, 1, 1);
    dim3 gridSizeS2(S2_SIZE, S2_SIZE, C1_KERNEL_DEPTH);
    subsample2D_kernel << <gridSizeS2, blockSizeS2 >> > (d_C1_output, C1_SIZE, d_S2_output, S2_SIZE);
    cudaDeviceSynchronize();

    // Step 3: Apply convolution (C3)
    dim3 blockSizeC3(C3_SIZE, 1, 1);
    dim3 gridSizeC3(C3_SIZE, C3_SIZE, C3_KERNEL_DEPTH);
    convolution2D_kernel << <gridSizeC3, blockSizeC3 >> > (d_S2_output, S2_SIZE, d_C3_output, C3_SIZE, C3_KERNEL_SIZE, d_C3_weights, d_C3_biases);
    cudaDeviceSynchronize();

    // Step 4: Apply subsampling (S4)
    dim3 blockSizeS4(S4_SIZE, 1, 1);
    dim3 gridSizeS4(S4_SIZE, S4_SIZE, C3_KERNEL_DEPTH);
    subsample2D_kernel << <gridSizeS4, blockSizeS4 >> > (d_C3_output, C3_SIZE, d_S4_output, S4_SIZE);
    cudaDeviceSynchronize();

    // Flatten S4 output (Step 5)
    flatten_kernel << <(C3_KERNEL_DEPTH * S4_SIZE * S4_SIZE + 255) / 256, 256 >> > (d_S4_output, C3_KERNEL_DEPTH, S4_SIZE, d_flattened_data);
    cudaDeviceSynchronize();

    // Step 5: Apply fully connected layer (F5)
    fully_connected_kernel << <(F5_SIZE + 255) / 256, 256 >> > (d_flattened_data, FLATTEN_SIZE, d_F5_output, F5_SIZE, d_F5_weights, d_F5_biases);
    tanh_kernel << <(F5_SIZE + 255) / 256, 256 >> > (d_F5_output, F5_SIZE);
    cudaDeviceSynchronize();

    // Step 6: Apply fully connected layer (F6)
    fully_connected_kernel << <(F6_SIZE + 255) / 256, 256 >> > (d_F5_output, F5_SIZE, d_F6_output, F6_SIZE, d_F6_weights, d_F6_biases);
    tanh_kernel << <(F6_SIZE + 255) / 256, 256 >> > (d_F6_output, F6_SIZE);
    cudaDeviceSynchronize();

    // Step 7: Apply fully connected layer (F7)
    fully_connected_kernel << <(F7_SIZE + 255) / 256, 256 >> > (d_F6_output, F6_SIZE, d_F7_output, F7_SIZE, d_F7_weights, d_F7_biases);
    softmax_kernel << <(F7_SIZE + 255) / 256, 256 >> > (d_F7_output, F7_SIZE);
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

    printf("\nAfter convolution (C3 data):\n");
    cudaMemcpy(C3_output, d_C3_output, C3_KERNEL_DEPTH * C3_SIZE * C3_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    TensorPrint(C3_output, C3_KERNEL_DEPTH, C3_SIZE, C3_SIZE);

    printf("\nAfter subsampling (S4 data):\n");
    cudaMemcpy(S4_output, d_S4_output, C3_KERNEL_DEPTH * S4_SIZE * S4_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    TensorPrint(S4_output, C3_KERNEL_DEPTH, S4_SIZE, S4_SIZE);

    printf("\nAfter fully connected layer (F5 output):\n");
    cudaMemcpy(F5_output, d_F5_output, F5_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    MatrixPrint(F5_output, F5_SIZE, 1);

    printf("\nAfter fully connected layer (F6 output):\n");
    cudaMemcpy(F6_output, d_F6_output, F6_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    MatrixPrint(F6_output, F6_SIZE, 1);

    printf("\nAfter fully connected layer (F7 output):\n");
    cudaMemcpy(F7_output, d_F7_output, F7_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    MatrixPrint(F7_output, F7_SIZE, 1);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_C1_weights);
    cudaFree(d_C1_output);
    cudaFree(d_C1_biases);
    cudaFree(d_S2_output);
    cudaFree(d_C3_weights);
    cudaFree(d_C3_output);
    cudaFree(d_C3_biases);
    cudaFree(d_S4_output);
    cudaFree(d_flattened_data);
    cudaFree(d_F5_weights);
    cudaFree(d_F5_output);
    cudaFree(d_F5_biases);
    cudaFree(d_F6_weights);
    cudaFree(d_F6_output);
    cudaFree(d_F6_biases);
    cudaFree(d_F7_weights);
    cudaFree(d_F7_output);
    cudaFree(d_F7_biases);
}

int main() {
    // Step 1: Iniatialiaz or load input
    printf("Loading image...\n");
    // initialize_input(); // Random input initialization
    load_image(FILENAME, imgIndex, HEIGHT, WIDTH);

    // Step 2: Print the original image
    printf("\nImage Before Padding:\n");
    create_and_print_image(HEIGHT, WIDTH);

    // Step 3: Pad the image to the required INPUT_SIZE
    printf("\nPadding the image...\n");
    pad_image(HEIGHT, WIDTH);

    // Step 4: Print the padded image
    printf("\nImage After Padding:\n");
    create_and_print_image(INPUT_SIZE, INPUT_SIZE);

    // Step 5: Initialize or load weights
    printf("\nInitializing weights...\n");
    // initialize_weights();  // Random weight initialization
    load_all_weights();
    
    // Step 6: Run the neural network on GPU
    printf("\nRunning the LeNet model on GPU...\n");
    run_lenet_gpu();
    
    return 0;
}
