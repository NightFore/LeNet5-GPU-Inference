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
        raw_data[i] = (float)rand() / RAND_MAX; // Valeur entre 0 et 1
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
        tensor[i] = (float)rand() / RAND_MAX;
    }
}

void convolution2D(float *input, float *output, float *kernels, int input_size, int kernel_size, int num_kernels, int output_size) {
    for (int k = 0; k < num_kernels; k++) {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_size; j++) {
                float sum = 0.0f;
                // Appliquer le noyau de convolution
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        // Calculer l'index de l'image et du noyau
                        int input_row = i + m;
                        int input_col = j + n;
                        // S'assurer que les indices sont dans les limites de l'image d'entrée
                        if (input_row < input_size && input_col < input_size) {
                            sum += input[input_row * input_size + input_col] * kernels[k * kernel_size * kernel_size + m * kernel_size + n];
                        }
                    }
                }
                // Stocker le résultat de la convolution dans le tableau de sortie
                output[k * output_size * output_size + i * output_size + j] = sum;
            }
        }
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
    int C1_data_height = 28; // 32 - 5 + 1 = 28
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
    
    // Perform the convolution
    convolution2D(raw_data, tensor_C1_data, tensor_C1_kernel, raw_data_size, kernel_size, C1_data_depth, C1_data_height);
    
    // Afficher la sortie de la convolution (tensor_C1_data)
    printf("Output of Convolution (C1 Data):\n");
    MatrixPrint(tensor_C1_data, C1_data_depth, C1_data_height, C1_data_width);  // Afficher la sortie de la convolution (6x28x28)
    
    // Perform the subsampling
    subsample2D(tensor_C1_data, tensor_S1_data, C1_data_height, S1_data_height, C1_data_depth);
    
    // Afficher la sortie du sous-échantillonnage (tensor_S1_data)
    printf("Output of Subsampling (S1 Data):\n");
    MatrixPrint(tensor_S1_data, S1_data_depth, S1_data_height, S1_data_width);  // Afficher la sortie du sous-échantillonnage (6x14x14)
    
    // Libération de la mémoire
    free(raw_data);
    free(tensor_C1_data);
    free(tensor_S1_data);
    free(tensor_C1_kernel);
    
}   
    
    
    
    
    
    
    
