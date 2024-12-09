#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Taille des matrices
#define RAW_DATA_SIZE 32
#define C1_DATA_SIZE 6
#define S1_DATA_SIZE 14
#define KERNEL_SIZE 5

// Fonction pour initialiser raw_data avec des valeurs aléatoires entre 0 et 1
void init_raw_data(float *raw_data, int size) {
    for (int i = 0; i < size * size; i++) {
        raw_data[i] = (float)rand() / RAND_MAX* 10.0f; // Range 0–10
    }
}

// Fonction pour initialiser C1_kernel avec des 0
void init_tensor_zero(float *tensor, int depth, int size) {
    for (int i = 0; i < depth * size * size; i++) {
        tensor[i] = 0;
    }
}

// Fonction pour initialiser C1_kernel avec des 0
void init_tensor_random(float *tensor, int depth, int size) {
    for (int i = 0; i < depth * size * size; i++) {
        tensor[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Range -1 to 1
    }
}

// CUDA device function to apply tanh activation
__device__ float activation_tanh(float M) {
    return tanhf(M);  // tanhf is a CUDA function for tanh
}

// CUDA kernel function to perform convolution
__global__ void convolution_kernel(float *input, float *output, float *kernels, int input_size, int kernel_size, int num_kernels, int output_size) {
    int k = blockIdx.z; // Kernel depth
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    if (i < output_size && j < output_size) {
        float sum = 0.0f;
        // Appliquer le noyau de convolution
        for (int m = 0; m < kernel_size; m++) {
            for (int n = 0; n < kernel_size; n++) {
                int input_row = i + m;
                int input_col = j + n;
                // S'assurer que les indices sont dans les limites de l'image d'entrée
                if (input_row < input_size && input_col < input_size) {
                    sum += input[input_row * input_size + input_col] * kernels[k * kernel_size * kernel_size + m * kernel_size + n];
                }
            }
        }
        // Appliquer la fonction d'activation tanh sur la sortie de la convolution
        output[k * output_size * output_size + i * output_size + j] = activation_tanh(sum);
    }
}

// Fonction pour effectuer le sous-échantillonnage 2D (moyenne de 2x2 pixels)
void subsample2D(float *input, float *output, int input_size, int output_size, int depth) {
    for (int k = 0; k < depth; k++) { // Loop through each channel (depth)
        for (int i = 0; i < output_size; i++) { // Loop through rows of the output
            for (int j = 0; j < output_size; j++) { // Loop through columns of the output
                float sum = 0.0f;
                // Average 2x2 block
                for (int m = 0; m < 2; m++) { // Loop over the 2 rows of the 2x2 block
                    for (int n = 0; n < 2; n++) { // Loop over the 2 columns of the 2x2 block
                        int input_row = 2 * i + m; // Corresponding row in the input
                        int input_col = 2 * j + n; // Corresponding column in the input
                        // Sum the values in the 2x2 block
                        sum += input[k * input_size * input_size + input_row * input_size + input_col];
                    }
                }
                // Store the average in the output
                output[k * output_size * output_size + i * output_size + j] = sum / 4.0f;
            }
        }
    }
}

// Fonction pour afficher une matrice 2D ou 3D (pour les tenseurs avec plusieurs canaux)
void MatrixPrint(float *M, int depth, int n, int p)
{
    for (int k = 0; k < depth; k++) { // Parcourir chaque canal
        printf("Channel %d:\n", k);
        for (int i = 0; i < n; i++) { // Parcourir les lignes
            for (int j = 0; j < p; j++) { // Parcourir les colonnes
                printf("%6.2f ", M[k * n * p + i * p + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    // Taille des matrices
    int raw_data_size = RAW_DATA_SIZE;
    int C1_data_depth = C1_DATA_SIZE;
    int C1_data_height = 28; 											// 32 - 5 + 1 = 28
    int C1_data_width = 28;
    int S1_data_depth = C1_DATA_SIZE;
    int S1_data_height = S1_DATA_SIZE;
    int S1_data_width = S1_DATA_SIZE;
    int kernel_size = KERNEL_SIZE;
    
    // Allocation de mémoire pour les matrices
    float *raw_data = (float*)malloc(raw_data_size * raw_data_size * sizeof(float));				// 32x32
    float *tensor_C1_data = (float*)malloc(C1_data_depth * C1_data_height * C1_data_width * sizeof(float)); 	// 6x28x28
    float *tensor_S1_data = (float*)malloc(S1_data_depth * S1_data_height * S1_data_width * sizeof(float)); 	// 6x14x14
    float *tensor_C1_kernel = (float*)malloc(C1_data_depth * kernel_size * kernel_size * sizeof(float)); 	// 6x5x5
    
    // Initialisations des variables
    init_raw_data(raw_data, raw_data_size);
    init_tensor_zero(tensor_C1_data, C1_data_depth, C1_data_height);
    init_tensor_zero(tensor_S1_data, S1_data_depth, S1_data_height);
    init_tensor_random(tensor_C1_kernel, S1_data_depth, kernel_size);
    
    // Afficher les données d'entrée (raw_data)
    printf("Raw Data (Input):\n");
    MatrixPrint(raw_data, 1, raw_data_size, raw_data_size);  // Afficher l'entrée (32x32)
    
    // Allouer la mémoire sur le GPU
    float *d_raw_data, *d_tensor_C1_data, *d_tensor_C1_kernel;
    cudaMalloc((void**)&d_raw_data, raw_data_size * raw_data_size * sizeof(float));
    cudaMalloc((void**)&d_tensor_C1_data, C1_data_depth * C1_data_height * C1_data_width * sizeof(float));
    cudaMalloc((void**)&d_tensor_C1_kernel, C1_data_depth * kernel_size * kernel_size * sizeof(float));
    
    // Copier les données sur le GPU
    cudaMemcpy(d_raw_data, raw_data, raw_data_size * raw_data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor_C1_kernel, tensor_C1_kernel, C1_data_depth * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor_C1_data, tensor_C1_data, C1_data_depth * C1_data_height * C1_data_width * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configurer la grille et les blocs
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((C1_data_height + 15) / 16, (C1_data_width + 15) / 16, C1_data_depth);
    
    printf("Kernels:\n");
    MatrixPrint(tensor_C1_kernel, C1_data_depth, kernel_size, kernel_size);
    
    // Lancer le kernel de convolution
    convolution_kernel<<<numBlocks, threadsPerBlock>>>(d_raw_data, d_tensor_C1_data, d_tensor_C1_kernel, raw_data_size, kernel_size, C1_data_depth, C1_data_height);
    
    // Copier la sortie du GPU vers la mémoire hôte
    cudaMemcpy(tensor_C1_data, d_tensor_C1_data, C1_data_depth * C1_data_height * C1_data_width * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Afficher la sortie de la convolution (tensor_C1_data)
    printf("Output of Convolution (C1 Data):\n");
    MatrixPrint(tensor_C1_data, C1_data_depth, C1_data_height, C1_data_width);  // Afficher la sortie de la convolution (6x28x28)
    
    // Perform the subsampling
    subsample2D(tensor_C1_data, tensor_S1_data, C1_data_height, S1_data_height, C1_data_depth);
    
    // Afficher la sortie du sous-échantillonnage (tensor_S1_data)
    printf("Output of Subsampling (S1 Data):\n");
    MatrixPrint(tensor_S1_data, S1_data_depth, S1_data_height, S1_data_width);  // Afficher la sortie du sous-échantillonnage (6x14x14)
    
    // Libération de la mémoire GPU et hôte
    cudaFree(d_raw_data);
    cudaFree(d_tensor_C1_data);
    cudaFree(d_tensor_C1_kernel);
    free(raw_data);
    free(tensor_C1_data);
    free(tensor_S1_data);
    free(tensor_C1_kernel);
    
    return 0;
}

    
    
    
    
    
